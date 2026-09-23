# faceback-45 — pre-labels, ready to import

armW (our production face detector) + CrowdHuman head-model corroboration, run over all 84,954 of
your frames. Nothing here needs a GPU any more — it is done.

## What is where

```
oss://we-vlm-annotation-data-sh/
├── faceback-45/images/                    your original upload, untouched (84,954)
├── faceback-45/prelabels/
│   ├── import_left.jsonl                  42,477 rows — feed this to face_view_import
│   ├── import_right.jsonl                 42,477 rows — same, right eye
│   ├── prelabels_full.jsonl               everything the detector produced (see "re-cutting")
│   └── README.md                          this file
├── faceback-45-left/images/               42,477 — renamed for the importer (see "why two prefixes")
└── faceback-45-right/images/              42,477
```

The two `faceback-45-{left,right}/` prefixes are **server-side copies** of your originals, byte for
byte (etag-verified on a 25-object sample). Your `faceback-45/images/` is untouched.

## Import

`tools/qa_review/face_view_import.py` on branch **`feat/face-view-import-allow-empty`**. Then the
three follow-up steps — **skipping the last one gives you a silently empty work page**:

```bash
for V in left right; do
  python tools/qa_review/face_view_import.py \
    --jsonl import_$V.jsonl --ds faceback-45-$V \
    --src "faceback45/armw_head@1" \
    --allow-empty --allow-off-grid --write

  python -m service.segment_import --ds faceback-45-$V --whole
  # create the review_job row, then:
  python -m service.work_item_import --job <job> --type frame.face
  python tools/portal_ops/supply_open.py --ds faceback-45-$V --job <job> \
    --types frame.face --by <you>
done
```

Two flags you need that were not on `dev`; both are in that branch:

- `--allow-empty` — keep frames the detector found nothing on, writing `boxes: []`. That is the
  "machine looked, no face here" state, which the portal shows differently from "no pre-label at
  all". 59% of these frames are empty, and they are where a genuine miss would hide, so dropping
  them would hide exactly what you want to find. (`--no-boxes` is a different thing: whole batch,
  no detector run, no `pii_box` rows at all.)
- `--allow-off-grid` — 13.5% of your frame indices are not multiples of 30. The importer's grid
  check is a self-check inherited from an earlier batch, not an engine invariant: `imgref.subject()`
  only requires `0..999999`. **Your data is regular** — every chunk steps by 60 — but 103 of 334
  sessions start at a non-zero phase (one begins at f8530, so all its frames are ≡10 mod 30). What
  the grid check actually protects against is left/right frame sets disagreeing; verified here,
  369/369 (session, chunk) pairs have **identical** left and right frame sets, including all 122
  that contain off-grid frames.

## The numbers

| | left | right |
|---|---:|---:|
| Frames | 42,477 | 42,477 |
| Boxes to review | 38,436 | 38,202 |
| Frames with no box | 25,015 (59%) | 25,091 (59%) |

**76,638 boxes total.** Frames are raw fisheye 2328×1748 — verified, 30/30 sampled. Coordinates in
the import files are already normalised 0..1 (top-left + w/h, 4 decimals, same convention as
`engine/worktype._num01`), so no conversion anywhere. Round-tripping them back to pixels agrees with
the source to within 0.216 px, i.e. under one quantisation step.

### The cut applied

A box is included if it clears **both**:

- `armw_score >= 0.75` **or** the head model corroborates it, **and**
- long side `>= 40 px`

The score-only view, for calibration:

| armW score | boxes | head-corroborated |
|---|---:|---:|
| ≥ 0.75 | 33,352 | 100% |
| 0.50–0.75 | 43,931 | 93% |
| 0.35–0.50 | 32,249 | 73% |
| 0.25–0.35 | 38,706 | **38%** |

Corroboration falls off a cliff below 0.35 — that bottom band is mostly texture (perforated
ceilings, shelf edges, moulding trays). That is why a bare low threshold is not usable.

### Re-cutting without re-running

`prelabels_full.jsonl` has **every** armW box down to 0.25 with its `armw_score`,
`long_side_px`, `head_xyxy`, `head_support` and `need_review`. Nothing was thrown away, so a
different threshold is a filter over that file, not another 85k-image inference pass. One row per
frame, in original pixel coordinates:

```json
{"key": "faceback-45/images/…_c001_left_f000000.jpg", "session": "…", "chunk": 1,
 "view": "left", "frame": 0, "width": 2328, "height": 1748,
 "n_boxes": 3, "n_need_review": 2,
 "boxes": [{"xyxy": [1334.2, 579.9, 1414.6, 681.4], "armw_score": 0.8213,
            "long_side_px": 101.5, "head_xyxy": [1321.0, 561.3, 1421.8, 690.2],
            "head_support": true, "need_review": true}]}
```

## Why two prefixes instead of one dataset

`face_view_import` derives each object's name as `<session>_c<chunk>_f<frame:06d}.jpg` — **no view
token** — and rejects any row whose `image` differs. Its own usage example is `--ds face-mine-right`:
the intended pattern is **one dataset per view**, with the ds prefix carrying left/right. Your
upload puts both views under one prefix with the view inside the filename, which those two
conventions cannot both satisfy — flattened into one ds the names would collide. Hence the copies.

## Things worth knowing before you look at the portal

- **`kind` stays `"work"` at round ≥ 1.** A review round is not `kind="review"` — that is a separate
  legacy slot model with an `{ok, fix}` answer shape, and it never receives `prev`. Tell rounds apart
  by the `round` field and the presence of `prev`. Filtering on `kind == "review"` returns nothing
  and looks exactly like "there is no review work".
- **`preferKind` is a response field, not a request knob.** You cannot ask for a review card.
- **A whole bench must be finished before it advances a round.** Answering part of a bench shows up
  in `work_round_avail` as "someone started", but the review candidate set stays empty
  (`work_item.bench_done_round` is still `-1`).
- **Review drafts DO prefill from the previous round.** The seed chain is
  `mine > prev > carried > pre` (`portal/src/worktypes.jsx`, `seedOf`). There was a real bug where
  it went straight to the machine pre-label and an untouched submit overwrote the annotator's work;
  fixed in `5b0ff28a` (2026-08-21), which is on `dev`. The fix also handles the subtle case where
  the previous round's answer is an empty array — `Array.isArray` rather than a truthiness test,
  because `[]` is falsy and would otherwise fall back to the machine boxes.
- Bench sizes here range from 10 to 180 items; the 369 (session, chunk) pairs are not uniform.

## Caveats on the data itself

- The frames are **unblurred faces of real people**. Same handling rules as any PII batch.
- `need_review` is our recommendation, not a verdict. Every box in `prelabels_full.jsonl` still
  carries its raw score, so disagreeing with the cut costs one pass over a 49 MB file.
- The head detector used for corroboration is **CrowdHuman YOLOv5m, AGPL-3.0** — internal data
  selection only, never a production path, not redistributed. It only decided which boxes get
  flagged; no head-model output is a label. `head_xyxy` rides along as evidence.
