"""LSP client for ``ty server`` running on the host.

Manages a persistent ``ty server`` subprocess via :mod:`subprocess`,
communicating using the Language Server Protocol (JSON-RPC over stdio with
``Content-Length`` framing).

The transport layer (framing, reader loop, request/response correlation)
is provided by :class:`~.lsp_base.LspBaseClient`.
This subclass implements the host-process launch/teardown using
``asyncio``-compatible pipes, plus ty-specific setup: installing ``ty`` if
absent, ensuring a ``pyproject.toml`` exists for module discovery, and
discovering site-packages so third-party imports resolve.
"""

import asyncio
import logging
import os
import shutil
import subprocess
from typing import Any

from .lsp_base import LspBaseClient, server_log_path

logger = logging.getLogger("metalgate_code")

_TY_INSTALL_TIMEOUT_SEC = 120
"""Timeout for installing ty if not present."""


class TyLspClient(LspBaseClient):
    """Persistent LSP client for ``ty server`` on the host.

    The server process is started lazily on first request and kept alive
    for the lifetime of the client.  All LSP requests are serialised through
    an :class:`asyncio.Lock` so that framing stays consistent.
    """

    def __init__(
        self,
        root_uri: str,
        *,
        python_path: str | None = None,
        venv_bin: str | None = None,
    ) -> None:
        super().__init__(root_uri)
        self._root_path = root_uri.replace("file://", "")
        self._python_path = python_path
        # Optional venv bin directory.  When set, ty is installed into and
        # launched from this venv so it uses the venv's Python and its
        # site-packages.  When unset, ty is discovered on PATH.
        self._venv_bin = venv_bin
        self._process: asyncio.subprocess.Process | None = None

    # ty setup

    async def _ensure_ty_installed(self) -> None:
        """Install ty if not present and ensure a pyproject.toml exists.

        ``ty`` needs a ``pyproject.toml`` (or ``ty.toml``) at the project
        root to discover first-party modules.  If none exists, we create
        a minimal one so that relative imports resolve correctly.

        When a venv is established (``venv_bin``), ty is installed into
        that venv so the ``ty`` binary lands in ``venv_bin`` and uses the
        venv's Python.  Otherwise it falls back to system pip.
        """
        # Check if ty is already available (in the venv or on PATH).
        ty_check_cmd = (
            f"{self._venv_bin}/ty --version 2>/dev/null"
            if self._venv_bin
            else "which ty 2>/dev/null"
        )
        exit_code, stdout = await self._shell(ty_check_cmd)
        if exit_code != 0 or not stdout.strip():
            logger.info("Installing ty…")
            if self._venv_bin:
                # Install into the venv so the ty binary lands in venv_bin.
                # uv-created venvs lack pip, so try pip first then uv pip.
                py = f"{self._venv_bin}/python"
                exit_code, stdout = await self._shell(
                    f"{py} -m pip install ty -q 2>&1",
                    timeout=_TY_INSTALL_TIMEOUT_SEC,
                )
                if exit_code != 0:
                    logger.info("pip not available in venv, trying uv pip install…")
                    exit_code, stdout = await self._shell(
                        f"uv pip install ty --python {py} -q 2>&1",
                        timeout=_TY_INSTALL_TIMEOUT_SEC,
                    )
            else:
                exit_code, stdout = await self._shell(
                    "pip install ty -q 2>&1", timeout=_TY_INSTALL_TIMEOUT_SEC
                )
            if exit_code != 0:
                raise RuntimeError(f"Failed to install ty: {stdout}")

        # Ensure pyproject.toml exists for module discovery.
        # Site-packages paths are passed via LSP initializationOptions
        # (see _customize_init_params), not via ty.toml — avoids writing
        # user files unless pyproject.toml is genuinely missing.
        root_path = self._root_path
        pyproject = f"{root_path}/pyproject.toml"
        try:
            exists = await asyncio.wait_for(self._fs_exists(pyproject), timeout=10)
        except (OSError, TimeoutError):
            exists = False

        if not exists:
            content = b'[project]\nname = "project"\nversion = "0.0.0"\n'
            try:
                await asyncio.wait_for(self._fs_write(pyproject, content), timeout=10)
            except (OSError, TimeoutError) as e:
                logger.warning("Failed to create pyproject.toml for ty: %s", e)

    async def _find_site_packages(self) -> list[str]:
        """Discover site-packages directories.

        Returns paths to the venv's site-packages so ty can resolve
        third-party imports.

        Uses the venv's Python when available, avoiding ``uv run`` which
        would create a second venv.  Falls back to system Python
        discovery only when no venv was established.
        """
        paths: list[str] = []

        # Prefer the venv's Python — no uv run, no second venv.
        if self._venv_bin:
            py = f"{self._venv_bin}/python"
            try:
                exit_code, stdout = await self._shell(
                    f"{py} -c 'import site; print(\"\\n\".join(site.getsitepackages()))'",
                )
                if exit_code == 0 and stdout.strip():
                    paths = [
                        p.strip() for p in stdout.strip().splitlines() if p.strip()
                    ]
            except OSError:
                logger.debug("venv site-packages discovery failed", exc_info=True)
            if paths:
                return paths

        # Fallback: system Python (no venv established).
        for cmd in (
            "uv run python -c 'import site; print(\"\\n\".join(site.getsitepackages()))'",
            "python -c 'import site; print(\"\\n\".join(site.getsitepackages()))'",
        ):
            try:
                exit_code, stdout = await self._shell(cmd)
                if exit_code == 0 and stdout.strip():
                    paths = [
                        p.strip() for p in stdout.strip().splitlines() if p.strip()
                    ]
                    if paths:
                        break
            except OSError:
                continue

        # Also look for a .venv site-packages under the project root
        try:
            exit_code, stdout = await self._shell(
                f"ls {self._root_path}/.venv/lib/ 2>/dev/null"
            )
            if exit_code == 0:
                for line in stdout.strip().splitlines():
                    name = line.strip()
                    if not name:
                        continue
                    candidate = f"{self._root_path}/.venv/lib/{name}/site-packages"
                    try:
                        exists = await asyncio.wait_for(
                            self._fs_exists(candidate), timeout=5
                        )
                        if exists and candidate not in paths:
                            paths.append(candidate)
                    except (OSError, TimeoutError):
                        pass
        except OSError:
            pass

        return paths

    async def _resolve_ty_command(self) -> str:
        """Return the ty command to launch (path or name)."""
        if self._venv_bin:
            ty_bin = f"{self._venv_bin}/ty"
            try:
                exists = await asyncio.wait_for(self._fs_exists(ty_bin), timeout=10)
            except (OSError, TimeoutError):
                exists = False
            if exists:
                return ty_bin
        ty_on_path = shutil.which("ty")
        if ty_on_path is not None:
            return ty_on_path
        raise FileNotFoundError("ty not found in venv or PATH")

    # Shared LSP initialization

    def _customize_init_params(self, params: dict[str, Any]) -> None:
        init_opts: dict[str, Any] = {}
        if self._python_path:
            init_opts["pythonPath"] = self._python_path
        log_file = server_log_path(self._root_uri, "ty")
        if log_file is not None:
            init_opts["logFile"] = log_file
            init_opts["logLevel"] = "debug"
        if init_opts:
            params["initializationOptions"] = init_opts

    # Shell / filesystem helpers (local)

    async def _shell(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> tuple[int, str]:
        full_env = dict(os.environ)
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=full_env,
        )
        try:
            stdout_bytes, _ = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout if timeout and timeout > 0 else None,
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return 1, ""
        return proc.returncode or 0, stdout_bytes.decode("utf-8", errors="replace")

    async def _fs_exists(self, path: str) -> bool:
        return os.path.exists(path)

    async def _fs_write(self, path: str, content: bytes) -> None:
        def _write() -> None:
            with open(path, "wb") as f:
                f.write(content)

        await asyncio.to_thread(_write)

    # Server lifecycle (local transport)

    async def _start_server(self) -> None:
        await self._ensure_ty_installed()

        # Discover site-packages so ty can resolve third-party imports.
        # Passed via PYTHONPATH so ty reads them as extra_paths.
        site_paths = await self._find_site_packages()
        env: dict[str, str] = {}
        if site_paths:
            env["PYTHONPATH"] = ":".join(site_paths)

        ty_cmd = await self._resolve_ty_command()

        full_env = dict(os.environ)
        if env:
            full_env.update(env)

        self._process = await asyncio.create_subprocess_exec(
            ty_cmd,
            "server",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # discard — no pipe to fill
            env=full_env,
        )

    async def _stop_server(self) -> None:
        if self._process is None:
            return
        # Closing stdin signals EOF to the LSP server, which exits cleanly
        # without needing os.kill (blocked by some sandbox profiles).
        if self._process.stdin is not None:
            try:
                self._process.stdin.close()
            except OSError:
                pass
        # Try a graceful terminate; tolerate PermissionError from sandboxes
        # that forbid os.kill — the stdin EOF above is the primary signal.
        try:
            self._process.terminate()
        except (ProcessLookupError, PermissionError):
            pass
        try:
            await asyncio.wait_for(self._process.wait(), timeout=5)
        except TimeoutError:
            try:
                self._process.kill()
            except (ProcessLookupError, PermissionError):
                pass
        self._process = None

    async def _on_stream_closed(self) -> None:
        if self._process is not None:
            if self._process.stdin is not None:
                try:
                    self._process.stdin.close()
                except OSError:
                    pass
            try:
                self._process.terminate()
            except (ProcessLookupError, PermissionError):
                pass
            self._process = None

    # Transport-specific read/write

    async def _write_raw(self, frame: bytes) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("ty server stdin is not available")
        self._process.stdin.write(frame)
        await self._process.stdin.drain()

    async def _read_raw(self) -> bytes | None:
        if self._process is None or self._process.stdout is None:
            return None
        try:
            data = await self._process.stdout.read(4096)
        except (OSError, asyncio.IncompleteReadError):
            logger.debug("error reading ty stdout", exc_info=True)
            return None
        if not data:
            return None
        return data

    # High-level LSP operations (python-specific override)

    def did_open(self, uri: str, text: str, language_id: str = "python") -> None:
        """Notify the server that a document was opened."""
        super().did_open(uri, text, language_id=language_id)
