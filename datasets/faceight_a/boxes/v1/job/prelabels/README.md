# faceight — pre-labels for verdict import
Source: /data/esteban/faceight/frames.csv rows with annotation_round=1 (42408 frames, all lview, 2328x1748, one chunk per session).
Detector: det_10g_armW (faceight_armW.jsonl), raw score floor ~0.1 -> prelabels_full.jsonl.
import_left.jsonl: boxes with score >= 0.5, normalized xywh (4dp); frames with none keep boxes=[] (--allow-empty).
Kept boxes 55951 of 506241; frames with >=1 box 23040; empty 19368.
Filenames renamed to importer convention <session>_c<chunk>_f<frame_idx:06d>.jpg; original t_ms filename kept as src_file in prelabels_full.jsonl.
