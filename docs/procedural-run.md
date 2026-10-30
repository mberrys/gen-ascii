# The procedural data run, start to delete

Generate a million procedural clouds (C2), use them in the model-size sweep (D1), keep everything
that records what happened, then delete the shards. The shards can be deleted because every cloud is
generated from its own seed: the same command at the same commit writes the same bytes. This run
proves that before it deletes anything.

Machine: RTX 3090 24 GB, 24 cores, 495 GB free on C:. Paths are relative to the repo root.

## What is kept and what is deleted

| item | size | after the run |
|---|---|---|
| `.cache/procedural/part-*/shard-*.points.npy` | ~49 GB | **deleted** |
| `.cache/procedural/part-*/meta.json`, `shard-*.jsonl` | ~215 MB | kept: what each row was, and the rebuild key |
| `.cache/procedural/deleted.json` | small | written: what was deleted and the exact rebuild commands |
| `.cache/training/objaverse` | 1.2 GB | kept: small, and every later step trains on it |
| `ml/runs/<run>/best.pt`, `train.json`, `train.log` | up to ~1 GB each | kept |
| `bench/results/<run>/` | small | kept and committed |

---

## 1. Code changes before generating anything

All six landed together, in one PR on top of `feat/procedural-data` (#66).

1. **Record the commit in `meta.json`.** `scripts/procedural-export.ts` writes the commit, any
   uncommitted source files and the sha256 of the bundled script. It refuses to start with
   uncommitted changes in `src/`, `scripts/`, `package.json` or `package-lock.json`, since rows from
   that state could not be rebuilt; `--allow-dirty` overrides it for a throwaway run, and the clean
   script then refuses to delete those shards.
2. **`--no-rgb` on the procedural export.** 156 of 200 procedural rows have no colour anyway, the
   loader never reads `rgb.npy`, and the colour head (D2) is a later step. Saves ~12 GB.
3. **More than one data folder in training.** `ml/cloudgen/train.py` takes one `--data`.
   - Make `--data` repeatable, with a `--mix` weight per folder (for example
     `--data .cache/training/objaverse --data .cache/procedural/part-* --mix 0.75,0.25`).
   - `CloudDataset.batch` draws each batch slot from a folder group by weight, then a row inside it.
     Without weights, a million procedural rows would drown 20,000 Objaverse rows at 98 to 2.
   - `train.json` records every folder's manifest: index, screening, commit and rows.
4. **Encode each unique caption once.** Procedural captions are templated, so a million rows hold
   only a few thousand distinct sentences. Encode the unique set and index into it, instead of
   running MiniLM a million times on the CPU.
5. **bf16 autocast in the training step.** The 3090 supports it. It roughly halves step time for the
   larger models, and checkpoints stay fp32.
6. **`scripts/procedural-clean.ts`**, the delete step in section 5 as code:
   - refuses unless every listed run's `train.json` has `finished`;
   - runs the rebuild check itself (5.2) and refuses unless it passes;
   - deletes only `*.points.npy` and `*.rgb.npy`;
   - writes `deleted.json`.

Tests for each: the mix weights in `ml/tests/test_data.py`, the commit refusal and `--no-rgb` in the
export's tests, and the clean script's refusals.

## 2. Pilot: 9,000 clouds

Shakes out the whole path at 1 per cent of the size before committing disk and GPU to it.

```sh
npm run export:procedural -- --out .cache/procedural/pilot --count 9000 --no-rgb
python -m cloudgen.train --data ../.cache/training/objaverse --data ../.cache/procedural/pilot \
  --mix 0.75,0.25 --out runs/pilot --steps 2000
```

Passes when:
- the loader reads 9,000 rows at 4,096 points with unit normals;
- `meta.json` carries the commit;
- 20 random clouds dropped into the app as `.ply` look like their captions (knots read as knots, and so on);
- training loss falls, and `train.json` lists both folders with their screening.

Then delete `.cache/procedural/pilot`.

## 3. Generate: a million clouds in eight processes

Build the bundle once, then run eight processes on separate seed ranges into separate folders. Don't
run `npm run export:procedural` eight times: each run rebuilds the same `.mjs` and they would
overwrite each other.

The export sweeps seeds item by item (nine items per seed). With `--count 125001` each process covers
exactly 13,889 seeds, so starting process `k` at seed `13889 * k` keeps the ranges apart. Total:
1,000,008 clouds.

```sh
npx esbuild scripts/procedural-export.ts --bundle --platform=node --format=esm --packages=external \
  --outfile=.cache/procedural-export.mjs
for k in 0 1 2 3 4 5 6 7; do
  node --max-old-space-size=4096 .cache/procedural-export.mjs \
    --out .cache/procedural/part-$k --count 125001 --seed $((13889 * k)) --no-rgb \
    > .cache/procedural/part-$k.log 2>&1 &
done
wait
```

- **Rate:** 200 clouds took 12 s in one process, so one process alone needs ~17 h. Eight should
  take ~2–3 h if disk keeps up.
- **Resumable:** a stopped process restarted with the same arguments skips the shards it already
  wrote.

**Check before training:**
- each folder has 126 `.jsonl` files and 125,001 rows;
- the eight `meta.json` files carry the same commit and different seeds;
- no uid appears in two folders (`cat part-*/shard-*.jsonl | jq -r .uid | sort | uniq -d` prints
  nothing);
- disk use is ~49 GB.

## 4. Train: the D1 sweep

All runs train on Objaverse (20k, screened) plus the procedural parts. Score each run's samples on
the 200 public prompts:

```sh
python -m cloudgen.sample --ckpt runs/<run>/best.pt --prompts ../bench/prompts.jsonl --limit 200 --steps 32 --out runs/<run>/samples
npm run score -- --samples ml/runs/<run>/samples
```

The held-out 200 are used once, on the final pick.

| run | model | data | steps | estimate on the 3090 |
|---|---|---|---|---|
| **R0a** | 37.7M (384 / 10 / 6) | Objaverse only | 30,000 | ~2 h (0.22 s a step, measured) |
| **R0b** | 37.7M | 75% Objaverse, 25% procedural | 30,000 | ~2 h |
| **R1** | ~100M (640 / 14 / 10) | the better of R0a and R0b | 60,000 | ~6–8 h with bf16 |
| **R2** | ~250M (1024 / 14 / 16) | same | 60,000 | ~15–20 h with bf16 |

The model column is width / depth / heads. The 100M and 250M sizes are approximate: use the count
`train.py` prints.

**R0a against R0b is the decision that matters.** It's the first full 30,000-step run; the baseline
stopped at 9,600. It answers whether procedural data helps at all:
- If R0b doesn't beat R0a on read rate, skip R1 and R2 on the mix, record the result, and go
  straight to section 5. The data has done its job by answering the question.
- If it helps, also try 50/50 at 37.7M before scaling, so the larger runs use the best mix.

**Pick the model:** the smallest within 2 points of the best on the public 200 (the plan's D1
rule). Score it once on the held-out 200 with `scripts/held-out.ts`, and record both numbers in
`docs/run-log.tsv` and `docs/training-model.md`.

## 5. Delete the shards

### 5.1 Check that nothing else needs them yet
The model-size sweep is done, but two later steps read training data again:
- **D3 distillation** trains a few-step student on data samples;
- **D4 fine-tuning** samples prompts, and may mix data back in.

If either starts right after this run, keep the shards until it finishes. Otherwise delete now: a
rebuild costs ~2–3 h of CPU and no GPU.

### 5.2 and 5.3 Prove the rebuild, then delete
Check out the commit the shards were written at (it is in every part's `meta.json`), then:

```sh
node scripts/procedural-clean.ts --root .cache/procedural --runs ml/runs/R0a,ml/runs/R0b,ml/runs/R1,ml/runs/R2 --dry-run
node scripts/procedural-clean.ts --root .cache/procedural --runs ml/runs/R0a,ml/runs/R0b,ml/runs/R1,ml/runs/R2
```

Before deleting anything it:
- refuses unless every run has finished, HEAD is the recorded commit, and the source has no
  uncommitted changes;
- rebuilds the export script and refuses unless it hashes to the build recorded in `meta.json`;
- regenerates the first shard of `part-0` and `part-7` and refuses unless the points and rows match
  the shards on disk byte for byte. If they don't, the data isn't reproducible: keep the shards and
  find out why.

Then it:
- writes `.cache/procedural/deleted.json` with the commit, the build command, the eight rebuild
  commands, the checksum of every array and the runs that read each part;
- deletes the `*.points.npy` (and any `*.rgb.npy`) under `part-*`;
- keeps `meta.json` and the `.jsonl` rows.

`train.json` in each run already names the commit and folders, so a future reader can tell exactly
what data trained each checkpoint and how to rebuild it.

### 5.4 Record it
Add a `decision` row to `docs/run-log.tsv`: the shards were deleted after the D1 sweep, the rebuild
was checked on two shards, and `deleted.json` holds the commands. Update `docs/procedural-data.md`
to say the data is generated on demand, not kept.

## Order and checkpoints

1. Code changes (section 1), merged
2. Pilot (section 2): passes, pilot deleted
3. Generate (section 3): row counts and uid check pass
4. R0a and R0b: **decision: does procedural data help?**
   - no → skip to 6
   - yes → mix check, then R1 and R2, then pick and held-out score
5. Results committed (reports, run log, model doc)
6. Delete (section 5): needs-check, rebuild proof, clean, record

At no point do two steps write into the same folder. The export refuses a folder holding a different
run, and the clean script refuses while a run is unfinished or the rebuild is unproven.
