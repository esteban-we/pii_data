# gt_bench_v1 — face-PII evaluation bench

The only honest measure this project has of detector recall. **Labels here must never be used for
training** — a model evaluated on data it trained on tells you nothing.

## Contents

```
labels/gt_bundle.json       FULL human labels: 9,636 frames / 20,054 boxes across 10 sequences
labels/gt_eval_sparse.json  reduced slice most evaluation scripts read: 889 frames / 1,541 boxes
images_sparse/              the 889 jpg that gt_eval_sparse.json refers to
```

Coordinate space: **raw fisheye 2328×1748, `vst_left`**. Boxes are `[x1, y1, x2, y2]` — absolute
pixel corners, not `x, y, w, h`.

`gt_bundle.json` structure: `{<chunk_uuid>: {"frames": {<frame_idx>: [[x1,y1,x2,y2], ...]}}}`.
Only frames containing at least one face are present. Sequences were labelled every 5th frame across
the whole chunk.

`gt_eval_sparse.json` is a list of `{path, boxes, session, chunk}`, produced by `gt_extract.py`:
120 frames per chunk, boxes under 40 px dropped (that filter removes only 54 of 20,054 boxes — the
frame sampling is what shrinks it).

## Two known biases in the sparse slice

Read these before quoting any recall number measured on it.

1. **It holds 941 of the 3,044 distinct faces (30.9%).** A face stays in view for ~6.6 labelled
   frames. Uniform sampling keeps the faces that linger and drops the ones that flash past — and the
   transient ones are the hard cases. Recall measured here is biased optimistic.
2. **Left eye only.** Production detects and blurs the two stereo views independently. Over 60
   chunks the detector behaves the same on both, but only 68% of candidate-bearing frames overlap
   and 16% of faces appear in the right eye alone. The quality conclusion holds; the coverage is
   half.

A replacement bench fixing both is being re-labelled — it is the `gt_bench` subset of
`face_pii/handoff_2026-08/`. When it returns, rebuild the sparse slice from it and **recompute every
historical recall figure**.

## Full frames

The 20,752 frames the full labels point at are not stored here: they are reproducible by decoding the
source videos in `we-fpv-sh-ns` at the frame indices in `gt_bundle.json` (`gt_extract.py` does this).
The labels are the irreplaceable part.
