# faceight_b: pre-labels for Verdict import (WOR-195)

Set: faceight_b = round 2 of `frames.csv` (rows with annotation_round == 2): 60,000 frames, both eyes (28,503 lview, 31,497 rview), one frame per line. Drawn in WOR-191 by cell shares over the 10 cells of `mining/faceight_cells.py` with `numpy.random.default_rng(19760703)`, breadth-first over episodes; round 2 is the first 60% of each cell's draw order, round 3 the rest. Frames are raw fisheye 2328 x 1748 jpg with unblurred faces of real people (PII): same handling rules as any PII batch.

Files: `import.jsonl` (one line per frame, the faceight round-1 / faceback-45 importer record: session, chunk "%03d", view lview / rview, frame_idx, width, height, boxes as normalized xywh with 4 decimals, image) and this README. OSS: `oss://we-vlm-annotation-data-sh/faceight_b/images/<image>` and `oss://we-vlm-annotation-data-sh/faceight_b/prelabels/{import.jsonl,README.md}`. The image name is `<session>_c<chunk>_f<frame_idx:06d>.jpg` with no eye token, as the faceback-45 importer files (`import_right.jsonl` there: view "rview", eye-less image) and round 1 (`faceight/`); the eye is the `view` field. Names are unique within this set and disjoint from the 42,408 faceight_a names (asserted by `mining/faceight_verdict_prep.py`). faceback-45 put each eye in its own dataset prefix (`faceback-45-left/`, `faceback-45-right/`); here both eyes share one prefix, so split `import.jsonl` by `view` if the importer wants one dataset per eye.

Prelabel rule: per frame exactly ONE detector, chosen by the frame's cell: armW only cells -> armW boxes (det_10g_armW); both and AA34 only cells -> armAA34 boxes (det_34g_armAA34); faceless -> no boxes (`boxes: []`, the "machine looked, no face here" state). Only that detector's boxes with score >= the cell's band edge are kept: 0.6+ cells -> 0.6, [0.3,0.6) cells -> 0.3, [0.1,0.3) cells -> 0.1. Nothing else is in the package: no second detector, no box under the edge, no scores. Boxes are converted as in round 1: x = x1 / width, y = y1 / height, w = x2 / width - x, h = y2 / height - y, clipped to [0, 1], 4 decimals, zero-area boxes dropped.

Cell rule (`mining/faceight_cells.py`, WOR-190): sW / sA = max box score of armW / armAA34 on the frame (0 without a box); band = band(max(sW, sA)) in 0.6+ / [0.3,0.6) / [0.1,0.3), faceless when neither detector has a box (every stored box is >= 0.1); within a band with lower edge lo: both = sW >= lo and sA >= lo, armW only = sW >= lo and sA < lo, AA34 only = sA >= lo and sW < lo.

## Counts per cell (total (lview / rview))

| cell | detector | edge | frames | boxes kept | frames with 0 boxes | zero-area dropped |
|---|---|---|---|---|---|---|
| armW only 0.6+ | det_10g_armW | 0.6 | 600 (228 / 372) | 668 (247 / 421) | 0 (0 / 0) | 0 (0 / 0) |
| both 0.6+ | det_34g_armAA34 | 0.6 | 2,400 (792 / 1,608) | 3,859 (1,278 / 2,581) | 0 (0 / 0) | 0 (0 / 0) |
| AA34 only 0.6+ | det_34g_armAA34 | 0.6 | 9,600 (4,592 / 5,008) | 10,120 (4,834 / 5,286) | 0 (0 / 0) | 0 (0 / 0) |
| armW only [0.3,0.6) | det_10g_armW | 0.3 | 1,200 (561 / 639) | 1,416 (687 / 729) | 8 (1 / 7) | 27 (8 / 19) |
| both [0.3,0.6) | det_34g_armAA34 | 0.3 | 4,800 (2,301 / 2,499) | 7,645 (3,607 / 4,038) | 0 (0 / 0) | 0 (0 / 0) |
| AA34 only [0.3,0.6) | det_34g_armAA34 | 0.3 | 19,200 (9,335 / 9,865) | 22,830 (11,090 / 11,740) | 0 (0 / 0) | 0 (0 / 0) |
| armW only [0.1,0.3) | det_10g_armW | 0.1 | 600 (297 / 303) | 1,725 (896 / 829) | 24 (12 / 12) | 454 (192 / 262) |
| both [0.1,0.3) | det_34g_armAA34 | 0.1 | 2,400 (1,188 / 1,212) | 5,299 (2,606 / 2,693) | 0 (0 / 0) | 0 (0 / 0) |
| AA34 only [0.1,0.3) | det_34g_armAA34 | 0.1 | 9,600 (4,622 / 4,978) | 15,418 (7,435 / 7,983) | 3 (1 / 2) | 3 (1 / 2) |
| faceless | none |  | 9,600 (4,587 / 5,013) | 0 (0 / 0) | 9,600 (4,587 / 5,013) | 0 (0 / 0) |
| **total** | | | 60,000 (28,503 / 31,497) | 68,980 (32,680 / 36,300) | 9,635 (4,601 / 5,034) | 484 |

## Sources (md5)

- `frames.csv`: bb56ee8f767a55239700f717f75361ed
- `round2_select.txt`: ccba075f6c4131ce93cfe9627e2a86df
- `faceight_armW.jsonl`: f9ee5193e5cb3e90234ec566eb56323f
- `faceight_armW_right.jsonl`: 82c9387f7b61aa57443c9537653f2c70
- `faceight_armAA34.jsonl`: 9041121f52fcb03230b66388ccf20089
- `faceight_armAA34_right.jsonl`: d1d9dbd87193950f32fe6cb1ff771607
- `round23_draw.csv`: 465727cc7ea5f3c93d822061e2acb2a1
- `verdict/import_left.jsonl`: 0462b3f322c723315884f3a5a0938290

Built by `mining/faceight_verdict_prep.py --round 2` (repo pii, WOR-195).
