# faceight_c: pre-labels for Verdict import (WOR-195)

Set: faceight_c = round 3 of `frames.csv` (rows with annotation_round == 3): 40,000 frames, both eyes (18,994 lview, 21,006 rview), one frame per line. Drawn in WOR-191 by cell shares over the 10 cells of `mining/faceight_cells.py` with `numpy.random.default_rng(19760703)`, breadth-first over episodes; round 2 is the first 60% of each cell's draw order, round 3 the rest. Frames are raw fisheye 2328 x 1748 jpg with unblurred faces of real people (PII): same handling rules as any PII batch.

Files: `import.jsonl` (one line per frame, the faceight round-1 / faceback-45 importer record: session, chunk "%03d", view lview / rview, frame_idx, width, height, boxes as normalized xywh with 4 decimals, image) and this README. OSS: `oss://we-vlm-annotation-data-sh/faceight_c/images/<image>` and `oss://we-vlm-annotation-data-sh/faceight_c/prelabels/{import.jsonl,README.md}`. The image name is `<session>_c<chunk>_f<frame_idx:06d>.jpg` with no eye token, as the faceback-45 importer files (`import_right.jsonl` there: view "rview", eye-less image) and round 1 (`faceight/`); the eye is the `view` field. Names are unique within this set and disjoint from the 42,408 faceight_a names (asserted by `mining/faceight_verdict_prep.py`). faceback-45 put each eye in its own dataset prefix (`faceback-45-left/`, `faceback-45-right/`); here both eyes share one prefix, so split `import.jsonl` by `view` if the importer wants one dataset per eye.

Prelabel rule: per frame exactly ONE detector, chosen by the frame's cell: armW only cells -> armW boxes (det_10g_armW); both and AA34 only cells -> armAA34 boxes (det_34g_armAA34); faceless -> no boxes (`boxes: []`, the "machine looked, no face here" state). Only that detector's boxes with score >= the cell's band edge are kept: 0.6+ cells -> 0.6, [0.3,0.6) cells -> 0.3, [0.1,0.3) cells -> 0.1. Nothing else is in the package: no second detector, no box under the edge, no scores. Boxes are converted as in round 1: x = x1 / width, y = y1 / height, w = x2 / width - x, h = y2 / height - y, clipped to [0, 1], 4 decimals, zero-area boxes dropped.

Cell rule (`mining/faceight_cells.py`, WOR-190): sW / sA = max box score of armW / armAA34 on the frame (0 without a box); band = band(max(sW, sA)) in 0.6+ / [0.3,0.6) / [0.1,0.3), faceless when neither detector has a box (every stored box is >= 0.1); within a band with lower edge lo: both = sW >= lo and sA >= lo, armW only = sW >= lo and sA < lo, AA34 only = sA >= lo and sW < lo.

## Counts per cell (total (lview / rview))

| cell | detector | edge | frames | boxes kept | frames with 0 boxes | zero-area dropped |
|---|---|---|---|---|---|---|
| armW only 0.6+ | det_10g_armW | 0.6 | 400 (133 / 267) | 457 (150 / 307) | 0 (0 / 0) | 0 (0 / 0) |
| both 0.6+ | det_34g_armAA34 | 0.6 | 1,600 (544 / 1,056) | 2,530 (849 / 1,681) | 0 (0 / 0) | 0 (0 / 0) |
| AA34 only 0.6+ | det_34g_armAA34 | 0.6 | 6,400 (3,041 / 3,359) | 6,734 (3,218 / 3,516) | 0 (0 / 0) | 0 (0 / 0) |
| armW only [0.3,0.6) | det_10g_armW | 0.3 | 800 (393 / 407) | 932 (474 / 458) | 1 (1 / 0) | 19 (15 / 4) |
| both [0.3,0.6) | det_34g_armAA34 | 0.3 | 3,200 (1,585 / 1,615) | 5,104 (2,468 / 2,636) | 0 (0 / 0) | 0 (0 / 0) |
| AA34 only [0.3,0.6) | det_34g_armAA34 | 0.3 | 12,800 (6,180 / 6,620) | 16,519 (7,911 / 8,608) | 0 (0 / 0) | 0 (0 / 0) |
| armW only [0.1,0.3) | det_10g_armW | 0.1 | 400 (187 / 213) | 1,121 (557 / 564) | 15 (6 / 9) | 294 (112 / 182) |
| both [0.1,0.3) | det_34g_armAA34 | 0.1 | 1,600 (761 / 839) | 3,558 (1,611 / 1,947) | 0 (0 / 0) | 0 (0 / 0) |
| AA34 only [0.1,0.3) | det_34g_armAA34 | 0.1 | 6,400 (3,070 / 3,330) | 10,476 (5,088 / 5,388) | 12 (5 / 7) | 12 (5 / 7) |
| faceless | none |  | 6,400 (3,100 / 3,300) | 0 (0 / 0) | 6,400 (3,100 / 3,300) | 0 (0 / 0) |
| **total** | | | 40,000 (18,994 / 21,006) | 47,431 (22,326 / 25,105) | 6,428 (3,112 / 3,316) | 325 |

## Sources (md5)

- `frames.csv`: bb56ee8f767a55239700f717f75361ed
- `round3_select.txt`: 55f10b3c67132d035b0574001d333902
- `faceight_armW.jsonl`: f9ee5193e5cb3e90234ec566eb56323f
- `faceight_armW_right.jsonl`: 82c9387f7b61aa57443c9537653f2c70
- `faceight_armAA34.jsonl`: 9041121f52fcb03230b66388ccf20089
- `faceight_armAA34_right.jsonl`: d1d9dbd87193950f32fe6cb1ff771607
- `round23_draw.csv`: 465727cc7ea5f3c93d822061e2acb2a1
- `verdict/import_left.jsonl`: 0462b3f322c723315884f3a5a0938290

Built by `mining/faceight_verdict_prep.py --round 3` (repo pii, WOR-195).
