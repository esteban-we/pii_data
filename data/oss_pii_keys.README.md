> Superseded by the data store indexes (PII-1413, 2026-09-23). The `oss_key` column of
> /data/esteban/pii/datasets/<name>/frames.csv is now the authoritative map from a
> file to its object, for all 13 sets; runs/train/MANIFEST.tsv does the same for the
> checkpoints. This file covers only the 8 sets of WOR-118.
>
> It is also stale in one respect: every `oss_key` line in data/oss_pii_keys.jsonl
> still reads `pii/<dataset>/...`, the pre-move key. The WOR-164 move itself did
> happen (a paginated listing on 2026-09-23 shows 192,701 objects under
> `pii/data/`, 100,008,675,734 B, and the 8 legacy prefixes gone), but the rewrite of
> this jsonl that the section below describes was never applied to the file in this
> repo. Read the keys from /data/esteban/pii, not from here.
>
> The `local_path` column is dead as well: it points into the tree now called
> /data/esteban/pii_backup, under a layout PII-1315 replaced. Nothing may resolve an
> image through it; an image is /data/esteban/pii/datasets/<dataset>/images/<image>.

# data/oss_pii_keys.jsonl

One line per object under `oss://algorithm-datasets/pii/data/`, produced by WOR-118
on 2026-09-09 and rewritten by WOR-164 on 2026-09-10 when every object was moved
from `pii/<dataset>/` to `pii/data/<dataset>/` (same basename, same bytes, same
md5; only the `oss_key` column changed). It is the mapping between the flat OSS keys and the local files under
`datasets/` in the tree now called `/data/esteban/pii_backup`, so a downloader could
rebuild that local tree and the flattened names could be reversed.

Each line is a JSON object with exactly these keys, sorted by `(dataset, oss_key)`:

| key | meaning |
|---|---|
| `dataset` | one of the eight prefix names below |
| `oss_key` | full object key, `pii/data/<dataset>/<basename>` |
| `local_path` | absolute path of the source file as of 2026-09-09, under the tree now called `/data/esteban/pii_backup` |
| `size` | bytes (equals the object's Content-Length) |
| `md5` | lowercase hex md5 of the file (equals the object's ETag, see below) |

## Layout on OSS

`pii/data/<dataset>/` per dataset, JPGs as immediate children, no `images/` level
and no session subdirectories. `pii/data/` sits beside `pii/models/` (checkpoints,
WOR-163). `face_pii/` and `siyu/` were not touched.

| dataset | local source | objects | bytes |
|---|---|---|---|
| face10k_v3 | face10k_v3/images | 8,339 | 4,438,714,792 |
| face10k_repair | face10k_v3/repair_v1/images | 3,168 | 2,093,367,062 |
| face_mine_v1 | face_mine_v1/images | 62,587 | 28,557,034,942 |
| faceback_45 | faceback_45/images | 84,954 | 56,159,427,855 |
| gt_bench_sparse | gt_bench_v1/images_sparse | 889 | 625,250,284 |
| gt_bench_full | gt_bench_v1/images_full | 9,636 | 3,201,611,182 |
| wider_face | wider_face/WIDER_train | 12,879 | 1,473,882,349 |
| pii_frames | pii_frames | 10,249 | 3,459,387,268 |
| total | | 192,701 | 100,008,675,734 |

## Flattening rules

Every rule was proven collision-free on disk before upload (`find | sort | uniq -d`
gave 0 duplicates in each case).

- face10k_v3, face10k_repair, face_mine_v1, faceback_45, gt_bench_sparse: already
  flat; basenames kept byte for byte. (face10k_v3 and face10k_repair use different
  filename conventions; that is expected, see docs/FACE_DATASETS.md section 3.)
- wider_face: `WIDER_train/images/<scene>/<name>.jpg` -> `<name>.jpg` (the scene
  directory is dropped; WIDER names embed the scene id already).
- pii_frames: `<session>/chunk_<NNN>/<view>/f<NNNNNN>.jpg` ->
  `<session>_c<NNN>_<view>_f<NNNNNN>.jpg`, view token exactly as on disk (`lview`
  or `rview`).
- gt_bench_full: `images_full/<session>_<chunk>/f<NNNNNN>.jpg` ->
  `<session>_<chunk>_f<NNNNNN>.jpg`. This set was not flat locally (raw basenames
  collide across chunks), so it takes the same naming that images_sparse already
  uses; the 889 sparse names are a subset of the 9,636 full names, but the bytes
  differ (separate encodes), so both prefixes are kept.

## Two deliberate count differences

- face10k_repair holds 3,168 images while the labelv2 manifest carries 3,152; all
  3,168 local images were uploaded (user decision: upload what is local).
- wider_face holds 12,879 local images while `face_pii/wider_face/` has 12,888
  objects; the 12,879 local images were uploaded (user decision: upload what is
  local, do not reconcile against face_pii).

## How it was produced

`data/oss_pii_upload.py` (stdlib only; OSS REST API with V1 signing; credential is
the `default` AK profile in `~/.aliyun/config.json`, region cn-shanghai, endpoint
oss-cn-shanghai.aliyuncs.com):

```
python3 data/oss_pii_upload.py upload --jobs 16      # idempotent, skips same-size keys
python3 data/oss_pii_upload.py count                 # remote count per prefix vs table
python3 data/oss_pii_upload.py assemble              # writes data/oss_pii_keys.jsonl
python3 data/oss_pii_upload.py sample --n 30         # HEAD 30 random objects per prefix
python3 data/oss_pii_upload.py verify-moved --require-source-empty   # listing vs jsonl, all prefixes
```

The upload went to `pii/<dataset>/`; the move to `pii/data/<dataset>/` (WOR-164,
2026-09-10) used the `copy`, `verify-moved` and `delete-moved` subcommands of the
same script: server-side CopyObject per key (`x-oss-copy-source`, nothing
re-uploaded, 16 parallel, resumable), then a full listing of the destination
asserting count and size/ETag == md5 for every record, then one DeleteObject per
verified source key (never a prefix delete), then a listing of the source asserting
0 objects; per dataset, stop on the first mismatch. The `aliyun` CLI listing was
taken as a second opinion on each count. CopyObject of a single-part object keeps
the ETag equal to the md5, and that was asserted on every copy response.

The run took 2,756 s wall for 191,812 objects (16 parallel PUTs; 9 to 50 MB/s
depending on file size, per-object latency bound on the small WIDER files) plus 67 s
for the 889 gt_bench_sparse objects at 12 parallel; 0 failed PUTs, 0 retries
surfaced. Remote counts were verified twice against the table with a paginated
listing, and 30 random objects per prefix were HEADed and compared to local
size/md5 (240/240 matched).

Every object was uploaded single-part with a `Content-MD5` header and the response
ETag was asserted equal to the local md5, so for this prefix `ETag == md5` for every
object (multipart uploads would break that; none were made, all files are < 1.2 MB).
`assemble` re-lists every prefix and checks each record's size and md5 against the
listing's Size and ETag before writing.

To re-derive a local tree from the file: for each line, download `oss_key` to any
root, keeping the path relative to `datasets/` in `/data/esteban/pii_backup`, then
compare size and md5. Do not write to `local_path` itself.
