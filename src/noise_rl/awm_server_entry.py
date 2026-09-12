"""Launch OpenEnv AWM with project-local, in-memory diagnostics enabled."""

from __future__ import annotations

import argparse

from .awm_server_diagnostics import install_awm_server_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local OpenEnv AWM server with diagnostics")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    install_awm_server_diagnostics()
    import uvicorn

    uvicorn.run("agent_world_model_env.server.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
