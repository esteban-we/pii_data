# faceight_c boxes v3: the second review pass over the same 9,925 frames

## Why

v2 was the armAE34 addition review. The same Verdict dataset `faceight_c_v2` was reviewed once
more and came back 2026-09-22 as `face_boxes_faceight_c_v3.jsonl` (md5
6e937bd530553b9cad5aa2175c416c67): the same 9,925 frames, `review_round` 2, the same prelabel
package (`machine_src` faceight_c_relabel_v1@1), 16,161 boxes. The user's 2026-09-23 rule is
that a frame present in a drop has its boxes FULLY REPLACED, not unioned; the user's 2026-09-30
rule is that v3 is the only faceight GT the project uses, which is why it is in the store at all
(PII-1681).

v3 is a full replacement over all 40,000 frames: the 9,925 reviewed frames take their boxes from
this pass, the other 30,075 keep their v1 rows byte for byte. Those 30,075 rows are v2's as
well, so v3 differs from v2 only inside the reviewed set. A consumer reads one version, never a
diff.

## Derivation

`data/build_faceight_pii2.py build --set c` reads `job/output/face_boxes_faceight_c_v3.jsonl`
and merges it over boxes/v1 on the drop's frames. That is the same merge
`training/merge_faceight_v3.py` in the fpv_face_pii repo made into the labelv2 manifest
`training/manifests/faceight_c_v3_labelv2.txt` (md5 5d090d821cebc5f929653a9fc79d2707), which is
where `eval_onnx.py --set faceight_c_v2` read its GT until PII-1681 repointed it here.

The prelabels are not copied a second time: they are v2's, byte for byte, under
`../v2/job/prelabels/`, and the build reads them from there (spec key `prelabel_version`) to
re-derive the reviewed frame set as the frames of `import_{left,right}.jsonl` that carry an
armAE34 addition. It asserts that set equals the drop's 9,925 frames.

Cross-check (PII-1681, `.knuth/tmp/pii1681/check_v3_vs_manifest.py` in the fpv_face_pii repo):
`boxes.csv` is box for box equal to that manifest on all 40,000 frames, compared as sorted
per-frame sets after rounding to one decimal with the frames mapped through the `local_name`
column of `frames.csv`. 0 frames differ; 33,763 boxes, 6,717 of them on eval frames, either way.
`ignore` is 0 on every row, as in v1 and v2. Coordinate convention as in v1.

## Counts

| version | boxes | images with a box | reviewed frames |
|---|---:|---:|---:|
| v1 | 28,042 | 15,896 | 40,000 (all) |
| v2 | 33,881 | 17,007 | 9,925 (the frames with an armAE34 addition) |
| v3 | 33,763 | 16,958 | 9,925 (the same frames, reviewed again) |

v3 has 27,046 boxes on the train frames and 6,717 on the eval frames. On the 9,925 reviewed
frames the return carries 16,161 boxes on 6,012 frames, against v2's 16,279 on 6,061 and the
10,440 v1 boxes those frames started with.

`boxes.csv` md5 e3268102de95c60449d24389029159e9.

## Files

- `job/output/`: byte copy of the drop
  `/data/esteban/tmp/pii_faceightr_v3/face_boxes_faceight_c_v3.jsonl`. There is no
  `job/prelabels/` here; see Derivation.
- `boxes.csv` and `frames.csv` are derived by `data/build_faceight_pii2.py`.

The written instruction sheet for this pass is not on file.
