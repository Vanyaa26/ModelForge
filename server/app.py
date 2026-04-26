"""FastAPI application for the ModelForge Environment.

Endpoints:
    - POST /reset: Reset environment (pick new dataset)
    - POST /step: Submit code, execute, return accuracy + reward
    - GET /state: Current environment state
    - WS /ws: WebSocket for persistent sessions
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with 'uv sync'"
    ) from e

try:
    from ..models import ModelforgeAction, ModelforgeObservation
    from .modelforge_environment import ModelforgeEnvironment
except (ModuleNotFoundError, ImportError):
    from models import ModelforgeAction, ModelforgeObservation
    from server.modelforge_environment import ModelforgeEnvironment


app = create_app(
    ModelforgeEnvironment,
    ModelforgeAction,
    ModelforgeObservation,
    env_name="modelforge",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
