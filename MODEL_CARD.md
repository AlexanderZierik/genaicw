# Part 2 model card

## Method and encoder choice

The final model uses rectified flow in the latent space of the frozen Stable
Diffusion 2 VAE (`Manojb/stable-diffusion-2-base`). The VAE maps each 112x112
RGB image to a 4x14x14 latent. A time-conditioned U-Net predicts the velocity
along the linear path between a training latent and Gaussian noise. At
generation time, `generate.py` integrates this learned ODE from noise to a
data latent with Heun's method, then decodes it through the frozen VAE.

Rectified flow was chosen because it is a modern latent generative method that
fits the coursework T4 budget and can be sampled with fewer function
evaluations than a conventional diffusion chain.

## Training data and preprocessing

Only the supplied 10,000 synthetic hot-dog training images are used. Each is
resized to 112x112, converted to RGB tensors in [0, 1], and encoded once with
the frozen VAE. The resulting 4x14x14 latents are cached locally under
`_data/` for repeated experiments; neither the raw images nor the cache are
committed.

## Budget actually spent

Complete this section from the JSON summaries emitted by
`train_flow_matching.py` after the GPU Dojo runs. Record the actual GPU, steps,
batch size, elapsed training time, peak memory, and total VM time.

| Run | Steps | OT | Public metric | Duplicate fraction | Training time | Result |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| Baseline | 20,000 | yes | pending | pending | pending | pending |
| Ablation | 10,000 | no | pending | pending | pending | pending |
| Final candidate | 40,000 | yes | pending | pending | pending | pending |

## What I tried and what happened

Complete this table only after evaluating each run with `selfeval.py` on a
fixed 64-image seed. Report before-and-after numbers and retain the negative
OT ablation if it performs worse. Select the final checkpoint by the public
metric together with visual diversity, not by classifier score alone.

## Final samples and known failure modes

Add the final 64-image contact sheet and the final loss curve after training.
Describe visible errors from the actual samples, such as malformed buns,
ambiguous toppings, repeated compositions, or background artefacts. Do not
claim realism that the figures do not support.

## External code, pretrained weights, and assistance

The pretrained Stable Diffusion 2 VAE is loaded from
`Manojb/stable-diffusion-2-base` through Hugging Face Diffusers. The supplied
course utilities provide the hot-dog data and public evaluation classifier.
The supplied coursework U-Net and rectified-flow reference were adapted for
reproducible training, cached latents, and documented experiment artifacts.
OpenAI Codex assistance was used to implement that engineering and draft this
model card. No external training images are used.
