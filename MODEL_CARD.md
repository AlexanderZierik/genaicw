# Model card

One page. Marked by a human, not by the challenge runner. An honest account of
what did *not* work earns marks; a list of claims without evidence does not.

## Method
What you built and why. Name the approach (latent diffusion, flow matching,
GAN, ...) and say what made you choose it.

## Training data and preprocessing
Which images, what resolution, what normalisation. Say so explicitly if you used
anything beyond the provided hot-dog data.

## Budget actually spent
Steps, batch size, wall-clock time, which GPU. Report what you really used, not
what you intended to use.

## What you tried, and what it did to your score
The important section. For each change: what you expected, what happened, and
the score before and after. Include the things that made it worse.

## Known failure modes
Where your model breaks. Look at your own samples and describe what is wrong
with them.

## External code and pretrained weights
Anything not written by you, including the provided VAE and utilities. Cite it.
