# faceight_b boxes v3: the second review pass over the same 9,402 frames

## Why

v2 was the armAE34 addition review. The same Verdict dataset `faceight_b_v2` was reviewed once
more and came back 2026-09-22 as `face_boxes_faceight_b_v3.jsonl` (md5
1b95adbf21a71061eaf58c4ee711d77d): the same 9,402 frames, `review_round` 2, the same prelabel
package (`machine_src` faceight_b_relabel_v1@1), 9,915 boxes. The user's 2026-09-23 rule is that
a frame present in a drop has its boxes FULLY REPLACED, not unioned; the user's 2026-09-30 rule
is that v3 is the only faceight GT the project uses, which is why it is in the store at all
(PII-1681).

v3 is a full replacement over all 60,000 frames: the 9,402 reviewed frames take their boxes from
this pass, the other 50,598 keep their v1 rows byte for byte. Those 50,598 rows are v2's as
well, so v3 differs from v2 only inside the reviewed set. A consumer reads one version, never a
diff.

## Derivation

`data/build_faceight_pii2.py build --set b` reads `job/output/face_boxes_faceight_b_v3.jsonl`
and merges it over boxes/v1 on the drop's frames. That is the same merge
`training/merge_faceight_v3.py` in the fpv_face_pii repo made into the labelv2 manifest
`training/manifests/faceight_b_v3_labelv2.txt` (md5 9d982b393a0e5bd2e89c8fc3933027aa), which is
where `eval_onnx.py --set faceight_b_v2` read its GT until PII-1681 repointed it here.

The prelabels are not copied a second time: they are v2's, byte for byte, under
`../v2/job/prelabels/`, and the build reads them from there (spec key `prelabel_version`) to
re-derive the reviewed frame set as the frames of `import_{left,right}.jsonl` that carry an
armAE34 addition. It asserts that set equals the drop's 9,402 frames.

Cross-check (PII-1681, `.knuth/tmp/pii1681/check_v3_vs_manifest.py` in the fpv_face_pii repo):
`boxes.csv` is box for box equal to that manifest on all 60,000 frames, compared as sorted
per-frame sets after rounding to one decimal with the frames mapped through the `local_name`
column of `frames.csv`. 0 frames differ; 62,507 boxes, 12,781 of them on eval frames, either
way. `ignore` is 0 on every row, as in v1 and v2. Coordinate convention as in v1.

## Counts

| version | boxes | images with a box | reviewed frames |
|---|---:|---:|---:|
| v1 | 65,267 | 24,514 | 60,000 (all) |
| v2 | 62,283 | 24,184 | 9,402 (the frames with an armAE34 addition) |
| v3 | 62,507 | 24,307 | 9,402 (the same frames, reviewed again) |

v3 has 49,726 boxes on the train frames and 12,781 on the eval frames. On the 9,402 reviewed
frames the return carries 9,915 boxes on 3,829 frames, against v2's 9,691 on 3,706 and the
12,675 v1 boxes those frames started with.

`boxes.csv` md5 317ee5e166c7ecb457893806772cde22.

## Files

- `job/output/`: byte copy of the drop
  `/data/esteban/tmp/pii_faceightr_v3/face_boxes_faceight_b_v3.jsonl`. There is no
  `job/prelabels/` here; see Derivation.
- `boxes.csv` and `frames.csv` are derived by `data/build_faceight_pii2.py`.

The written instruction sheet for this pass is not on file.
