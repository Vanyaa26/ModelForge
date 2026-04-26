"""FastAPI application for the ModelForge Environment.

Endpoints:
    - POST /reset: Reset environment (pick new dataset)
    - POST /step: Submit code, execute, return accuracy + reward
    - GET /state: Current environment state
    - WS /ws: WebSocket for persistent sessions
"""

try:
    from fastapi.responses import HTMLResponse
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


@app.get("/", response_class=HTMLResponse)
@app.get("/web", response_class=HTMLResponse)
def web():
    """Small landing page for Hugging Face Spaces."""
    return """
    <!doctype html>
    <html>
      <head>
        <title>ModelForge</title>
        <style>
          body { font-family: system-ui, sans-serif; max-width: 760px; margin: 48px auto; line-height: 1.55; }
          code { background: #f3f3f3; padding: 2px 5px; border-radius: 4px; }
          a { color: #2563eb; }
        </style>
      </head>
      <body>
        <h1>ModelForge</h1>
        <p><strong>An OpenEnv environment where an LLM learns to train ML models.</strong></p>
        <p>
          Call <code>POST /reset</code> to receive a dataset description, then submit
          Python training code to <code>POST /step</code>. The environment executes the
          code and returns accuracy, reward, and training diagnostics.
        </p>
        <ul>
          <li><a href="/docs">API docs</a></li>
          <li><a href="/health">Health check</a></li>
          <li><a href="https://github.com/Vanyaa26/ModelForge">GitHub source</a></li>
        </ul>
      </body>
    </html>
    """


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
