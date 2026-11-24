"""Verify a listening packaged ACE backend without trusting its HTTP response."""

import json
import os
import sys

import psutil

_ENTRYPOINT = ("-I", "-B", "-c", "from openjarvis.cli import main; main()", "--quiet")
_MODES = {
    "--local-services-only": "local_services",
    "--local-inference-only": "inference",
}
_CLOUDFLARE_ARGS = (
    "--engine",
    "cloudflare",
    "--model",
    "@cf/google/gemma-4-26b-a4b-it",
    "--agent",
    "simple",
)


def _canonical(path):
    return os.path.normcase(os.path.realpath(path))


def _config_path(environ):
    return _canonical(
        environ.get("OPENJARVIS_CONFIG")
        or os.path.join(environ["OPENJARVIS_HOME"], "config.toml")
    )


def _verified_server(port):
    from openjarvis.core.paths import get_config_dir

    profile = get_config_dir()
    with (profile / "server.json").open(encoding="utf-8") as file:
        state = json.loads(file.read(4096))
    with (profile / "server.pid").open(encoding="utf-8") as file:
        registered_pid = int(file.read(32))
    if (
        type(state) is not dict
        or type(state.get("pid")) is not int
        or state["pid"] <= 0
        or state.get("host") != "127.0.0.1"
        or state.get("port") != port
        or state.get("ready", True) is not True
        or registered_pid != state["pid"]
    ):
        return None

    process = psutil.Process(state["pid"])
    if _canonical(process.exe()) != _canonical(sys.executable):
        return None
    # Keep the process identity exact across native JSON round trips.
    started_at = process.create_time().hex()
    args = process.cmdline()
    expected = (*_ENTRYPOINT, "serve", "--host", "127.0.0.1", "--port", str(port))
    if len(args) < len(expected) + 2 or tuple(args[1 : 1 + len(expected)]) != expected:
        return None
    managed_args = tuple(args[1 + len(expected) :])
    if managed_args == _CLOUDFLARE_ARGS:
        mode = "cloudflare"
    else:
        mode = _MODES.get(args[-1])
        if mode is None:
            return None
    environ = process.environ()
    home = environ.get("OPENJARVIS_HOME")
    if (
        not home
        or _canonical(home) != _canonical(profile)
        or _config_path(environ) != _config_path(os.environ)
    ):
        return None
    if not any(
        conn.status == psutil.CONN_LISTEN and conn.laddr == ("127.0.0.1", port)
        for conn in process.net_connections(kind="tcp4")
    ):
        return None
    if not process.is_running():
        return None
    return process, {"pid": process.pid, "started_at": started_at, "mode": mode}


def main():
    port = int(sys.argv[1])
    try:
        verified = _verified_server(port)
    except (KeyError, ValueError, OSError, psutil.Error):
        verified = None
    if len(sys.argv) == 2:
        print(json.dumps(None if verified is None else verified[1]))
        return 0

    # An explicit frontend restart may stop the exact adopted process. The
    # verification above is repeated, and psutil checks PID reuse at termination.
    expected = json.loads(sys.argv[2])
    if verified is None:
        return 0
    process, identity = verified
    if identity != expected:
        return 2
    try:
        process.terminate()
        process.wait(timeout=8)
    except psutil.NoSuchProcess:
        pass
    except psutil.Error:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
