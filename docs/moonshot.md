# What landed, against the plan

The moonshot plan is a program for a team of fourteen over four stages. This is what one agent could
land in a session: the opening sequence the plan itself says goes first, the two dependencies with
the most steps waiting on them, and two of the distribution paths. Every number here came from a
command in this repository that can be rerun.

| plan step | landed in | state |
|---|---|---|
| 0.1 a 200-prompt benchmark with a vision judge | #58 | measured, see `docs/bench.md` |
| 0.2 a first server-side baseline model | #59 | trained and sampled; read rate in the run log |
| 0.3 the demo: replay a trajectory as the assembly | #59 | `?traj=` in the app, frames in `docs/shots/trajectory/` |
| B2 split a DOM-free core | #60 | `src/core/`, `snapshotText` and the sampler under Node |
| B4 a trajectory API on the stage | #59 | `stageApi.stream`, stated in `docs/contracts.md` |
| I1 an ANSI renderer | #61 | truecolor, half block, diffed frames |
| I2 the CLI | #62 | `npm run cli -- teapot` |
| A1 1,000 public prompts and 200 sealed held-out | this PR | `bench/prompts.jsonl`, `bench/held-out.jsonl.enc`, `scripts/held-out.ts --check` |
| A3 scoring a training run's own samples | this PR | `bench/score.ts`, one warm tab, `score.json` a loop can read |

Not attempted, with the dependency that blocks each:

| plan step | why not |
|---|---|
| A2, A4, A5 (a local judge, legibility metrics, a scheduled run) | A2 needs A1's prompts rendered and judged first, which the run just committed makes possible; A4 and A5 are the next week of harness work |
| B1, B3 (profile, WebGPU) | needs real-device traces and three's TSL port; nothing in this session could verify a 60 fps claim |
| C1 to C4 (larger pool, procedural data, teacher pipeline, re-captioning) | each is a batch job measured in hours of downloads and model calls |
| D1 to D6 (scale, colour, distillation, medium-aware tuning, ONNX, hosting) | needs the 100M to 250M parameter sweeps on a cluster, and D5's quantisation needs the student D3 produces |
| E1 to E6 (the runtime in the browser) | needs D5's ONNX files; the app has the trajectory side of E2 already |
| F1 to F6 (glyph descriptors, G-buffer, structure-matched glyphs) | needs B3's compute pass |
| G, H, J, K (motion, calligram, safety output checks, gallery) | each is its own stage, and stage 3 and 4 are gated on the model |

## The numbers

200 prompts, 12,000 glyphs, dark theme, one machine.

| source | reads as its prompt | thumbnail | wall clock p50 |
|---|---|---|---|
| search sources | 39% (78/200) | 35% | not recorded |
| scene writer | 50% (99/200) | 60% | not recorded |
| the baseline model | 0% (0/200) | 1% | 6.2 s |

The judge agrees with a human eye on 36 of 40 images (90 per cent), and every disagreement was the
judge being kinder than the eye, so the read rates are a little generous.

The baseline reading zero is the honest starting point: 9,600 steps on one card buys a model that
draws a glyph-shaped mass, not a teapot. The plan says nobody expects it to win yet, and the number
says how far it has to go. `docs/bench.md` holds the per-category tables and the commands.

## What a reader should take from it

The benchmark is the deliverable that lasts. A model is a number that changes; the instrument that
says whether the next model is better is the thing every later step needs, and it is small enough to
run in an hour on one machine.
