# The model

A point cloud from a prompt, small enough to run in a browser tab, trained on the app's own export.
This is the baseline the rest of the plan has to beat, not the model the plan wants: it is one
evening on one card.

```sh
python -m cloudgen.train --data <export folder> --out runs/base --steps 30000
python -m cloudgen.sample --ckpt runs/base/last.pt --prompts bench/prompts.jsonl --steps 32 --out runs/base/samples
```

## The trade the plan is built on

A glyph render needs 4,096 points with normals, not a photoreal mesh. General text-to-3D models
cannot make that trade because their output has to survive close inspection; this one only has to
survive being drawn in letters. Everything below follows from that.

## Shape

| | |
|---|---|
| points | 4,096, each xyz and a unit normal |
| tokens | 256 patches of 16 points, so a token carries 96 numbers |
| width / depth / heads | 384 / 10 / 6 |
| parameters | 37.7M |
| conditioning | a frozen MiniLM sentence embedding (`all-MiniLM-L6-v2`, 384-d), both added to the time embedding and used as one cross-attention memory token |
| transport | rectified flow: `x_t = (1-t) x_0 + t x_1`, target `x_1 - x_0`, Euler at sampling |

Rectified flow rather than epsilon prediction because the app draws the walk. The straight-line
target means the sampling path is short and legible, and every frame of it is a target the stage's
springs can chase. A 32-step walk is 33 frames including the noise the object assembles out of.

## Training

| | |
|---|---|
| data | 19,980 clouds from `.cache/training/objaverse` (`docs/training-data.md`) |
| batch | 32 clouds |
| optimiser | AdamW, lr 1e-4 cosine with 500 warmup steps, weight decay 0.01 |
| augmentation | a random turn about Y per sample, because the export keeps each file's own facing |

The run that shipped: 9,600 steps of a 30,000-step cosine schedule, 0.22 s a step on one RTX 3090,
stopped because the measurement mattered more than the last third of the schedule. Its read rate on
the benchmark is 0 of 200 at full size and 1 per cent at thumbnail size (`docs/bench.md`). The record
a trained model quotes is `train.json` beside its weights: data, screening, config, minutes, loss.

## The walk is the animation

`ml/cloudgen/sample.py` writes two files per prompt:

- `<id>.traj`, every frame `[frames][4096][6]`, which the app's `trajectory` source replays through
  `stageApi.stream` (`docs/contracts.md`). Frame 0 is the noise, so the first thing on screen is the
  dust the object is about to become.
- `<id>.ply`, the final frame as a point cloud, which the app's own file loader reads and the
  benchmark scores with `--sources ply`.

## Known gaps

- **Scale.** 37.7M parameters and 20k clouds is a baseline, not the model the plan's D1 sweep asks
  for (40M/100M/250M on C1+C2+C3 data). The read rate is the number that says how far it has to go.
- **No colour head.** D2 is a later step, so a sampled cloud has no colour and falls back to the
  palette the way a library object does.
- **No distillation.** 32 steps is not a browser budget; D3's few-step student is what makes E1's
  runtime fit.
- **One card, one process.** No EMA weights, no checkpoint sweep, no seed averaging.
