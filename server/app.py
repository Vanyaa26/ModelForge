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
          * { box-sizing: border-box; }
          body {
            margin: 0;
            min-height: 100vh;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: #e5f7ff;
            background:
              radial-gradient(circle at 20% 20%, rgba(0, 229, 255, 0.18), transparent 30%),
              radial-gradient(circle at 80% 10%, rgba(168, 85, 247, 0.18), transparent 28%),
              linear-gradient(135deg, #020617 0%, #08111f 45%, #030712 100%);
            display: grid;
            place-items: center;
            padding: 32px;
          }
          .card {
            width: min(900px, 100%);
            padding: 44px;
            border: 1px solid rgba(125, 211, 252, 0.25);
            border-radius: 28px;
            background: rgba(2, 6, 23, 0.72);
            box-shadow: 0 0 70px rgba(34, 211, 238, 0.12), inset 0 0 40px rgba(255,255,255,0.03);
            backdrop-filter: blur(16px);
          }
          .eyebrow {
            color: #67e8f9;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.22em;
            text-transform: uppercase;
          }
          h1 {
            margin: 14px 0 12px;
            font-size: clamp(42px, 8vw, 86px);
            line-height: 0.95;
            letter-spacing: -0.07em;
          }
          .glow {
            color: #fff;
            text-shadow: 0 0 28px rgba(103, 232, 249, 0.45);
          }
          p {
            max-width: 720px;
            color: #b6c8d8;
            font-size: 18px;
            line-height: 1.7;
          }
          .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px;
            margin: 30px 0;
          }
          .metric {
            padding: 18px;
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 18px;
            background: rgba(15, 23, 42, 0.72);
          }
          .metric strong {
            display: block;
            color: #ffffff;
            font-size: 24px;
          }
          .metric span { color: #8aa4b8; font-size: 14px; }
          .links { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 28px; }
          a {
            color: #07111f;
            background: linear-gradient(135deg, #67e8f9, #a78bfa);
            padding: 11px 16px;
            border-radius: 999px;
            font-weight: 700;
            text-decoration: none;
          }
          code {
            color: #a5f3fc;
            background: rgba(8, 47, 73, 0.7);
            padding: 3px 7px;
            border-radius: 7px;
          }
        </style>
      </head>
      <body>
        <main class="card">
          <div class="eyebrow">OpenEnv Hackathon India 2026</div>
          <h1><span class="glow">ModelForge</span></h1>
          <p>
            A reinforcement-learning environment where an LLM learns to train ML models.
            The agent receives a dataset description, writes Python training code, and gets
            rewarded from real execution metrics: accuracy, improvement, efficiency,
            recovery, and overfit control.
          </p>
          <p>
            Inspired by Karpathy's AutoResearch loop, ModelForge adds the missing piece:
            the model that runs experiments is also updated through SFT and DPO.
          </p>
          <section class="grid">
            <div class="metric"><strong>7</strong><span>classification datasets</span></div>
            <div class="metric"><strong>14/14</strong><span>DPO evaluation success</span></div>
            <div class="metric"><strong>0.588</strong><span>best average reward</span></div>
          </section>
          <p>
            Use <code>POST /reset</code> to get a dataset and <code>POST /step</code>
            to submit training code.
          </p>
          <div class="links">
            <a href="/docs">API Docs</a>
            <a href="/health">Health</a>
          </div>
        </main>
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
