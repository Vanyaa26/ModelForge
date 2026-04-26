# ModelForge Pitch Video Script

Target length: under 2 minutes.

## Shot Plan

- 0:00-0:20: Face camera hook.
- 0:20-1:25: Screen recording of README, environment reset/step, notebook training curve, and sample generated code.
- 1:25-1:55: Face camera close.

## Script

What if I told you an AI model could learn to become a better ML engineer, not by reading more tutorials, but by actually trying experiments and updating its own weights?

In 2026, Andrej Karpathy's AutoResearch showed a powerful direction: let an AI agent run research experiments, measure results, and keep what works. That was inspiring. But one thing stayed fixed: the agent itself did not learn. It could run experiments overnight, but the model behind it was still the same model.

That is where ModelForge starts.

ModelForge is an OpenEnv environment where an LLM gets a real dataset description: samples, features, classes, class balance, baselines, and example rows. Its action is to write Python training code. The environment runs that code, trains the model, evaluates it on a held-out test split, and returns accuracy plus a reward.

The reward is not just "did accuracy go up?" It combines five signals: final accuracy, improvement over baseline, efficiency, recovery from errors, and overfitting control. So the agent is being trained to behave less like a random code generator and more like a practical ML engineer.

For training, we used Qwen 2.5 1.5B with TRL. First we run the base model in the environment. Then we do SFT on successful training scripts. Then DPO teaches the model to prefer code that worked over code that crashed or underperformed.

Here is the key result: across seven datasets, the average accuracy improved from 0.846 for the base model, to 0.875 after SFT, and 0.929 after DPO. The notebook includes the real reward curves, accuracy curves, and training loss plots.

What I like about this environment is that it is not just testing if an LLM can write sklearn code once. It tests whether a model can learn the habit of choosing the right approach for the data in front of it.

That is ModelForge: an OpenEnv environment for training models that learn how to train models.

## Screen Recording Checklist

- Show the Hugging Face Space link from the README.
- Show `env.reset()` returning dataset metadata.
- Show `env.step(code)` returning accuracy, reward, and reward breakdown.
- Show `AutoLearn_SUBMIT.ipynb` training cells for SFT and DPO.
- Show the result table or `autolearn_results.png`.
- End on the README with links to the Space, notebook, and video.
