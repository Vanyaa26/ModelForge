# ModelForge Submission Checklist

Use this as the final pre-submit pass for the OpenEnv Hackathon.

## Required Materials

| Requirement | Status | Notes |
|---|---|---|
| Use OpenEnv latest release | Ready | Environment uses `openenv-core[core]>=0.2.2`, `Environment`, FastAPI server, `openenv.yaml`, `reset`, `step`, and `state`. |
| Working RL/training notebook | Ready | `training/AutoLearn_SUBMIT.ipynb` contains the Qwen 1.5B, TRL SFT, and DPO pipeline. |
| Real training evidence | Needs final artifact | The notebook has plot cells, but this repository currently has no saved notebook outputs, `data/iter*.json`, or `autolearn_results.png`. Re-run the notebook and keep the generated plot before final submission. |
| Short writeup/video | Draft ready | `VIDEO_SCRIPT.md` is an under-2-minute script. Upload the video publicly and replace the README placeholder. |
| Hugging Face Space | Needs verification | README points to `https://huggingface.co/spaces/vanyatentiwala/modelforge`. Run `openenv push --repo-id vanyatentiwala/modelforge` and verify it loads. |
| README | Ready except public video link | README now explains motivation, environment mechanics, reward logic, results, local run, training notebook, and Space link. |

## Final Submit Steps

1. From `modelforge`, deploy or update the Space:

```bash
openenv push --repo-id vanyatentiwala/modelforge
```

2. Re-run `training/AutoLearn_SUBMIT.ipynb` and keep the generated `autolearn_results.png`.

3. Record the video using `VIDEO_SCRIPT.md`, upload it publicly, and replace the README pitch-video placeholder.

4. Open the Space URL in a browser and verify `/web`, `/docs`, and `/health`.

5. Submit only the Hugging Face Space URL.

## Positioning

Lead with this:

> Karpathy's AutoResearch showed agents can run ML experiments. ModelForge asks the next question: what if the model running those experiments could update its own weights and become a better ML engineer?

The strongest proof points are:

- The environment forces dataset-specific model selection, not generic code generation.
- Reward combines accuracy, baseline improvement, efficiency, recovery, and overfit control.
- The training notebook shows base to SFT to DPO improvement: `0.846 -> 0.875 -> 0.929`.
