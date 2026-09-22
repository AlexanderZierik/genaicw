# Model card: Part 1, your own VAE

One page, marked by a human. This is the Part 1 report and the filename is not
negotiable: it is fetched from your marked commit by exact name and rendered
into the PDF your marker reads. `report_part1.md` is not collected and the
marker sees an empty section.

Blurry blobs on the hot-dog stage are the **correct** result. Marks come from
understanding the failure, not from avoiding it. The staff reference scores
0.00 / 25 on the leaderboard.

## Architecture and parameter count

Encoder, reparameterisation, decoder. State the parameter count and show it is
**under 1 million** (print it, do not estimate).

## Likelihood choice, for each stage, and why

Not "the reconstruction loss". Say what you assumed about the data and what that
implies: Bernoulli for MNIST, Gaussian for hot dogs, and what you had to change
about normalisation and the output activation to match. Refer to a histogram of
your own pixel values.

## Training, both stages

Beta and any annealing schedule, optimiser, learning rate, epochs, and the
**actual** wall-clock time and which GPU. If you hit posterior collapse, say so
and show the two loss terms separately.

## Figures

Committed under `figures/` and referenced from this file, so they are rendered
into the marker's PDF:

- loss curves, the two terms plotted separately
- reconstructions beside their inputs
- samples from the prior
- an interpolation between two encoded images
- the MNIST t-SNE, coloured by digit class

A figure you generated but did not commit appears in the marker's PDF as
"not found in the repository at this commit".

## Leaderboard entry

`figures/leaderboard.png`, that exact path: a screenshot of **your Part 1 row**,
showing that commit's hash and its score. This is the only way your Part 1
commit can be identified at marking time, because LabTS records only the commit
you mark at the deadline. Leave the file in the repository.

## Your analysis

Why MNIST works and hot dogs do not, in your own words, referring to your own
images. The interesting part is not that it failed but *which property of the
model class* made it fail, and why the same network succeeded on the other
dataset.
