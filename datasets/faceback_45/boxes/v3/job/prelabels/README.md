# faceback_hq re-label prelabels v1 (PII-946)

The current faceback_hq GT plus the armAE34 boxes that GT does not cover, so the labelers
judge only the additions. No inference was run: this is a filter over the existing
per-frame dump.

## Source

- dump: `/home/esteban/repos/pii/.knuth/pages/media/face-mine-eval/faceback_hq/armAE34.json` (16,940 frames, 2,213,293 detections down to score 0.02)
- model: armAE34 (`/data/esteban/pii/runs/train/armAE34_det34g.onnx`), det_size 1024, coords in original 2328x1748 px
- GT: pii-data dataset `faceback_hq` boxes `v1` as the dump carries it: `gt`
  (19,047 boxes, long side >= 40 px, what the eval scores) plus `gt_ignored`
  (3,969 boxes, under 40 px; faceback_hq has no ignore-flagged boxes, so
  gt_ignored is exactly the sub-floor ones)
- images: `/data/esteban/pii/datasets/faceback_45/images/<file>`; faceback_hq is a relabel of
  the faceback_45 eval frames (PII-176) and owns no bytes of its own

## The cut

A detection is an ADDITION when both hold:

- `score >= 0.3`;
- max IoU against EVERY GT box of that frame, `gt` and `gt_ignored` together, is `< 0.1`.

No size floor: tiny boxes are kept. IoU is plain intersection over union on the dump's
pixel corners (one decimal, the precision of the pii-data source table). Boxes are not
clipped to the frame, following faceback_45; 436 additions reach outside.

## Counts

| | |
|---|---:|
| Frames in the dump | 16,940 |
| Detections at score >= 0.3 | 28,788 |
| Additions (after the IoU gate) | 6,713 |
| Frames with at least one addition | 4,023 |
| Sessions covered | 63 |
| lview / rview frames | 2,110 / 1,913 |
| GT boxes carried | 8,348 (6,359 eval + 1,989 ignored) |
| Of those frames, with no GT at all | 1,515 |

Additions by score band:

| band | boxes |
|---|---:|
| 0.3-0.4 | 3,926 |
| 0.4-0.5 | 1,818 |
| 0.5-0.6 | 734 |
| 0.6-0.8 | 232 |
| 0.8+ | 3 |

Additions by long side (px):

| band | boxes |
|---|---:|
| <20 | 17 |
| 20-40 | 3,573 |
| 40-60 | 1,207 |
| 60-100 | 1,201 |
| 100+ | 715 |

## Files

`additions.jsonl`: one row per frame that has at least one addition, in the dump's frame
order. Original pixels.

```json
{"file": "<session>_c<chunk>_<left|right>_f<idx>.jpg", "session": "...",
 "width": 2328, "height": 1748,
 "boxes": [{"xyxy": [1334.2, 579.9, 1414.6, 681.4], "score": 0.8213}]}
```

`import.jsonl`: the same frames in the faceback_45 prelabel shape (`datasets/faceback_45/prelabels/README.md`), normalised top-left x,y,w,h at 4 decimals.
GT first in source order, then the additions in the dump's score-descending order.

```json
{"session": "...", "chunk": "032", "view": "lview", "frame_idx": 0,
 "width": 2328, "height": 1748, "image": "<session>_c<chunk>_f<idx:06d>.jpg",
 "boxes": [{"x": 0.5731, "y": 0.3317, "w": 0.0175, "h": 0.0295, "source": "gt"},
           {"x": 0.7204, "y": 0.5449, "w": 0.0233, "h": 0.0494, "source": "gt", "gt_ignored": true},
           {"x": 0.2, "y": 0.3, "w": 0.02, "h": 0.03, "source": "armAE34", "score": 0.4123}]}
```

`source` is `gt` or `armAE34`; machine boxes carry `score`; a GT box that the eval ignores
(long side under 40 px) carries `gt_ignored: true`. `image` drops the view token because
`face_view_import` derives `<session>_c<chunk>_f<idx:06d>.jpg` and rejects any other name;
the view lives in the `view` field and in the per-view dataset prefix, as faceback_45 did.

`stats.json`: every count above, machine readable.

## Re-cutting at another score or IoU

Nothing was thrown away: the dump holds every detection down to 0.02 with its score, plus
both GT lists, so another cut is one pass over that file and needs no GPU.

```sh
/data/esteban/pii/runs/eval_venv/bin/python mining/faceback_hq_relabel_v1.py build \
  --out-dir <dir> --score-cut 0.4 --iou-gate 0.1
/data/esteban/pii/runs/eval_venv/bin/python mining/faceback_hq_relabel_v1.py check \
  --out-dir <dir> --score-cut 0.4 --iou-gate 0.1
```

Measured cuts (additions / frames with an addition), same IoU gate 0.1:

| score | additions | frames |
|---|---:|---:|
| 0.3 | 6,713 | 4,023 |
| 0.4 | 2,787 | 2,022 |
| 0.5 | 969 | 854 |

Raising the gate to 1.0 disables it: every detection at the score becomes an addition
(28,788 at 0.3), which is the sabotage check that the gate is not vacuous.

## Handling

These are unblurred faces of real people. PII: this directory stays on this box; nothing
goes into the repo and nothing goes to an external service.
