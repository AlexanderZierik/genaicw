# figures/

Commit the images your model cards reference here, and the leaderboard
screenshot, which has a fixed name:

    figures/leaderboard.png     <- your leaderboard row (Part 1, criterion P5)

Everything in this directory that a model card references with
`![](figures/whatever.png)` is pulled from your marked commit and rendered into
the PDF your marker reads, so a broken path means the marker sees the filename
instead of the picture. `figures/leaderboard.png` is fetched by name whether or
not a model card references it, and gets its own page.

Images do not count towards the model-size part of your challenge score, so
committing figures never costs you marks.
