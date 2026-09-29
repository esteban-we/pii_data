# faceight

faceight picks 1/8 of the episode corpus at random, with the same scene mix as the
corpus, from episodes not yet in any train/eval set.

- `mining/faceight_sample.py`: draws the sample (seed 19760703); writes the files below.
- `data/episode_usage.csv`: one row per episode: where it is used, scene category, `faceight` 0/1,
  `n_chunks` and `duration_s` (sum of chunk durations) probed from OSS by
  `mining/episode_chunks_probe.py` (WOR-62); -1 / empty = not probed or errored;
  `chunk_sel` = index of the latest chunk >= 240 s (WOR-66), -1 = none.
- `data/faceight/episodes.txt`: the 34,667 picked episode ids, frozen.
- `data/faceight/residues.csv` (`s,used,batch,view`): which seconds (0..239) of the selected
  chunk we take a frame at (`t_ms = 1000 * s + 233`) and on which eye (`view`: `vst_left` or
  `vst_right`; key (s, view), so a second eye on the same second is a second row); flip `used`
  under a new `batch` to grow. `faceight_pull.py --view V` selects only the rows with
  `view == V` and exits if there are none (WOR-103).
- `mining/faceight_pull.py`: pulls each frame from OSS by time (snapshot at `t_ms`, key
  `<session>/chunk_<chunk_sel:03d>/vst_left/vst_left_video.mp4`) to
  `shang:/data/esteban/faceight/frames/<session>_c<chunk_sel:03d>_lview_t<ms>.jpg`, then
  runs armW (`mining/rview_detect.py`, det_size 1024). Episodes with `chunk_sel` -1 go to
  `failed.jsonl` (`no-chunk-240`). Resumable; `--limit N` = first N episodes.

# Frame record schema

`shang:/data/esteban/faceight/faceight_armW.jsonl`, one JSON line per fetched frame:

    episode_id, session_id, chunk (= chunk_sel), view (vst_left), s, batch, t_ms (1000*s+233),
    frame_idx, w, h, file, model, n_faces, boxes

- `frame_idx`: nominal index from the mp4 stts (frame whose pts is nearest to `t_ms`);
  the snapshot API is addressed by time, so this is derived, not returned by OSS.
- `n_faces`: boxes with score >= 0.6. `boxes`: `[x1, y1, x2, y2, score]` in raw pixels,
  all detections with score >= 0.1 after NMS, score descending.
- `model`: onnx stem, e.g. `det_10g_armW`.
- Frames whose selected chunk is missing in OSS, or shorter than `t_ms`, get no record;
  they are listed with a reason in `failed.jsonl` next to the output.

# Batch 1 result (2026-09-09, WOR-60)

Residues {3, 7, 9} (batch 1) and {60, 120, 180} (batch 2) pulled in one run: 186,084 frames = 31,014
episodes x 6, 0 fetch failures, 3,653 episodes skipped (`no-chunk-240`). At score >= 0.6: 87.96% of frames
have 0 faces, 5.81% one, 6.23% two or more (0.236 faces/frame; 6.07 boxes/frame at >= 0.1). Files:
`shang:/data/esteban/faceight/{faceight_armW.jsonl,fetch.jsonl,failed.jsonl,frames/}` (105 GiB of jpg).
`frame_idx` follows the mp4 stts; on 32 videos with irregular stts it differs from round(fps*t_ms) by up to 16.

# Batches 3 to 5 result (2026-09-12, WOR-74)

Each eye now has 12 residues: batch 3 {30, 90, 135, 150, 210, 225} and batch 4 {15, 45, 75, 105, 165, 195}
on vst_right, batch 5 {22, 52, 82, 112, 142, 172} on vst_left; two runs in parallel (`--view vst_right
--batch 4` and `--view vst_left --batch 5`). Left: 186,084 new frames, 0 failures, 372,168 records in
`faceight_armW.jsonl`. Right: 372,144 new frames; 24 are missing because two right videos do not exist in OSS
(`20260616_005222_UFQJYN/chunk_028`, `20260722_094131_NHBSLF/chunk_016`, HTTP 404, the same two as round 1),
36 frames on 3 videos needed a recovery rerun after a DNS error; 414,545 records in `faceight_armW_right.jsonl`
(42,401 round-1 twins at s in {3, 7, 9, 60, 120, 180} plus the new residues). At score >= 0.6 the new frames
have 88.1% 0 faces, 5.8% one, 6.0% two or more on either eye (0.229 faces/frame; 6.13 left and 5.99 right
boxes/frame at >= 0.1), the same as batches 1 to 2. Disk: frames/ 210 GiB, frames_right/ 234 GiB (588 KB/frame).
Records of batches 1 to 2 and of round 1 are byte-identical to before. frames.csv has these frames as rows of
their own since WOR-106 (one row per frame of either eye, `view` column).

# armAA34 boxes (2026-09-12, WOR-185)

A second detector over the same frames, no fetch, on shang: `faceight_pull.py --stage detect --model
shang:/data/esteban/pii/runs/train/armAA34_det34g.onnx --model-name det_34g_armAA34 --out-jsonl ...` (SCRFD-34G,
armAA34 of WOR-146, decode checked exact against evaluation/eval_onnx.py on 20 frames). Same record schema,
`model` = `det_34g_armAA34`, the frame fields copied from the same fetch record as armW's. Files:
`shang:/data/esteban/faceight/faceight_armAA34.jsonl` (372,168 records, all of frames/: batches 1, 2, 5) and
`faceight_armAA34_right.jsonl` (372,144 records, batches 3 and 4 only). The 42,401 round-1 right twins in
frames_right/ (s in {3, 7, 9, 60, 120, 180}) are not scored: round 1 is annotated externally and round-1 frames
are excluded from later selection. Run as 3 shards per eye on 6 GPUs (`--shard K/N`, `--done-jsonl`; the
shard files `faceight_armAA34*.shard{0,1,2}.jsonl` are kept next to the merged ones), 49 min, 42 fps per shard.
At score >= 0.6 armAA34 gives 86.6% 0 faces, 6.7% one, 6.7% two or more on either eye (0.250 left / 0.248
right faces per frame; 2.98 / 2.97 boxes per frame at >= 0.1) against armW's 88.0 / 5.8 / 6.1% (0.233 / 0.229;
6.10 / 5.99 boxes). Per frame, n_faces agrees with armW on 93.7% of frames, armAA34 higher on 4.2 / 4.3%,
lower on 2.1 / 2.0% (left / right). `--out-jsonl` is the file the resume reads, so a rerun of every command
reports 0 to do and leaves the files unchanged.

# Right-view frames for round 1 (WOR-96)

The 42,408 round-1 rows of frames.csv also have their right-eye frame: same session, chunk and
`t_ms`, key `<session>/chunk_<chunk:03d>/vst_right/vst_right_video.mp4`, file
`<session>_c<chunk:03d>_rview_t<ms>.jpg`. Pulled with `mining/faceight_pull.py --view vst_right
--frames-from /data/esteban/faceight/frames.csv --round 1` (the CSV rows replace the residue rule;
each row's chunk is checked against `chunk_sel`); with `--view vst_right` every output gets a
`_right` suffix (`--out-suffix` overrides): `shang:/data/esteban/faceight/{frames_right/,
fetch_right.jsonl, failed_right.jsonl, timing_right.json, faceight_armW_right.jsonl}`. Records have
the schema above with `view` = vst_right and `file` = the rview name; `frame_idx` comes from the
right video's own stts (it is a separate mp4). frames.csv carries two extra columns, `file_right`
and `n_faces_right` (empty for rows without a right frame), appended by
`mining/faceight_round.py --merge-right /data/esteban/faceight/faceight_armW_right.jsonl`; the merge
copies every other column byte for byte and is rerunnable.

Result (2026-09-10): 42,401 of the 42,408 round-1 frames pulled (26.96 GB, 636 KB/frame); 7 have no
right video in OSS (HTTP 404 on the moov: `20260722_094131_NHBSLF/chunk_016`, all 6 residues, and
`20260616_005222_UFQJYN/chunk_028`, its one round-1 frame), listed in `failed_right.jsonl`. At score
>= 0.6 the right frames have 52.5% 0 faces, 20.9% one, 26.6% two or more (0.955 faces/frame; 11.74
boxes/frame at >= 0.1) against 47.2 / 25.5 / 27.3% (1.037) on the same left frames; 79.3% of the
pairs have equal n_faces, mean |difference| 0.234. `frame_idx` differs from round(fps*t_ms) by 3..5
on 11 frames (10 videos with irregular stts, same as the left set). frames.csv md5
1fe1cf081e55b51be0028e90cbf8e344 on all three copies.

# VLM head count (WOR-70)

`mining/faceight_vlm.py` sends every frame in `fetch.jsonl` (downscaled to 1024 px long side) to
Qwen2.5-VL-72B-Instruct-AWQ served by vLLM on shang (`mining/faceight_vlm_server.sh`), prompt
`heads_v1`: "Count the human heads visible in this image. Count every real person's head in any
orientation (facing the camera, facing away, seen from above or from the side), including heads
that are only partly visible or cut off by the image border. Do not count mannequins, dolls,
posters, photos, screens, statues or reflections. Answer with a single integer and nothing else."
Output `shang:/data/esteban/faceight/faceight_vlm.jsonl`, one JSON line per frame:

    file, n_heads (int, null if the answer was not an integer), raw (verbatim answer),
    model, prompt_id, latency_ms

Progress: `vlm_progress.json` and `vlm.log` next to it; `mining/faceight_vlm_status.sh` prints
both from the local machine. Resumable (files already in the output are skipped).

# frames.csv (WOR-94)

git tracks this file gzipped, as `data/faceight/frames.csv.gz` (8.6 MB): the plain CSV is
107,370,665 B, over GitHub's 100 MB blob limit, and is gitignored (PII-1639). Every reader
still names the `.csv` and opens it through `pii_root.open_index`, which takes the plain file
when it is on disk and the `.gz` otherwise. After `faceight_round.py` rewrites the CSV, refresh
the tracked copy with `gzip -9 -n -c data/faceight/frames.csv > data/faceight/frames.csv.gz`.

`data/faceight/frames.csv` (also `shang:/data/esteban/faceight/frames.csv`), one row per pulled
frame, built by `mining/faceight_round.py` from `faceight_armW.jsonl`: `file, episode_id,
session_id, chunk, s, batch, t_ms, frame_idx, n_faces, annotation_round`. Boxes are not copied;
they stay in the JSONL on shang (frames are PII, only names and counts travel). Rows sorted by
`file`. Round 1 (42,408 rows) = every frame with `n_faces >= 1` (22,408) plus 20,000 frames drawn
uniformly without replacement from the 163,676 zero-face frames with
`numpy.random.default_rng(19760703)` over the zero-face file names sorted; everything else is
`annotation_round` 0. Append-only: rerunning keeps every nonzero round and only fills rows still
at 0; `--round N --select-file LIST` assigns a later round to listed rows still at 0 (no built-in
rule beyond round 1). Run on shang with `/data/esteban/armw/venv/bin/python` (stdlib + numpy).
When residues.csv grows, `mining/faceight_round.py --append` adds the new JSONL frames as rows at
`annotation_round` 0 (empty `file_right`, `n_faces_right`) at the end of the file, sorted by `file` among
themselves; every existing line is kept byte for byte and a frames.csv row missing from the JSONL is an
error (WOR-95). File order is therefore not a key: the round-1 draw is frozen, later rows never change it,
and the built-in round-1 rule refuses to fill rows once round 1 exists (use `--select-file`). `--dry-run`
reports the count without writing.
Since WOR-106 the table is one row per frame of either eye, keyed by `file` (the rview name for right
frames), with two more columns at the end: `view` (`vst_left` / `vst_right`, same values as residues.csv) and
`n_faces_aa34` (armAA34 count at score >= 0.6 from `faceight_armAA34.jsonl` / `faceight_armAA34_right.jsonl`,
WOR-185). Header: `file, episode_id, session_id, chunk, s, batch, t_ms, frame_idx, n_faces, annotation_round,
file_right, n_faces_right, view, n_faces_aa34`. The columns were appended line by line
(`--add-view-aa34 --aa34-jsonl ...`): every byte of the 186,084 earlier rows is unchanged (the file with the
two columns stripped has the WOR-96 md5 1fe1cf081e55b51be0028e90cbf8e344), so the round-1 draw (42,408 left
rows at `annotation_round` 1, uploaded to Verdict) is frozen. `--append` takes `--view {vst_left,vst_right}`
and the armW JSONL of that eye plus `--aa34-jsonl`; a right record whose file equals an existing row's
`file_right` (the 42,401 round-1 twins) is skipped and reported, not appended, so `file_right, n_faces_right`
stay the only twin link and only on round-1 rows. Result (2026-09-12): 744,312 rows = 186,084 (batches 1, 2)
+ 186,084 left appended (batch 5) + 372,144 right appended (batches 3, 4; 24 frames of two 404 videos
missing), `n_faces_aa34` filled on every row, 701,904 rows at `annotation_round` 0; md5
b2acba9162ecf4088937b918f27f960e on all three copies. `--round N --select-file` now edits the
`annotation_round` field of the listed lines in place (no re-serialisation, no JSONL needed); a table with
the `view` column is never re-serialised. Reruns of every mode are no-ops; an `--add-view-aa34` rerun keeps
values the given JSONLs do not cover and refuses to change a value already present.
Rounds 2 and 3 rule (WOR-191, 2026-09-12). Pool: the 701,904 rows at `annotation_round` 0 (both eyes,
frames.csv md5 b2acba9162ecf4088937b918f27f960e). Each pool frame is put in exactly one of the 10 cells of
`mining/faceight_cells.py` (WOR-190): sW / sA = max box score of armW / armAA34 in the JSONLs (0 without a
box); band = band(max(sW, sA)) in 0.6+ / [0.3,0.6) / [0.1,0.3), faceless when neither detector has a box;
within a band with lower edge lo, both = sW >= lo and sA >= lo, armW only = sW >= lo and sA < lo, AA34 only =
sA >= lo and sW < lo. Shares (percent of N = 100,000), bands 0.6+ / [0.3,0.6) / [0.1,0.3): armW only 1 / 2 / 1,
both 4 / 8 / 4, AA34 only 16 / 32 / 16, faceless 16; quota = share x 1,000 (1,000 / 2,000 / 1,000, 4,000 /
8,000 / 4,000, 16,000 / 32,000 / 16,000, 16,000). Draw per cell with one `numpy.random.default_rng(19760703)`
consumed in that (class-major) order: the cell's episode_ids sorted and shuffled once; pass 1 takes one
uniformly chosen unused frame of each episode in that order until the quota, pass 2 a second frame from each
episode that still has one, and so on (breadth-first; both eyes pooled). Within each cell the first 60% of the
draw order is round 2 and the rest round 3 (60,000 and 40,000 frames). Only the two AA34-only cells exceed
their episode count (5,838 and 11,897 episodes for 16,000 and 32,000 frames: 8 passes, up to 8 frames of one
episode); every other cell is one frame per episode. Realised: round 2 = 60,000 frames from 25,118 episodes
(28,503 left, 31,497 right), round 3 = 40,000 from 18,704 (18,994 left, 21,006 right), together 28,753 of the
31,014 pool episodes; scene_category mix within 3 points of the faceight corpus. Script
`mining/faceight_round2_select.py`, report `reports/faceight_round2_draw.md`; select files `round2_select.txt`,
`round3_select.txt` and the audit trail `round23_draw.csv` (file, round, cell, pass, episode, session, view,
sW, sA) next to this README and in `shang:/data/esteban/faceight/`. Assigned with `faceight_round.py --round 2
--select-file` then `--round 3 --select-file` (line-level, dry run first, 0 rows already nonzero); frames.csv
md5 bb56ee8f767a55239700f717f75361ed after, on all three copies, counts {0: 601,904, 1: 42,408, 2: 60,000,
3: 40,000}, round-1 rows and every other byte unchanged.
Verdict packages for rounds 2 and 3 (WOR-195, 2026-09-12): faceight_b = the 60,000 round-2 rows, faceight_c = the
40,000 round-3 rows, both eyes in one file, built by `mining/faceight_verdict_prep.py --round {2,3}` (stdlib +
numpy, imports `faceight_cells.cell_of`; `--dry-run`, `--out-dir`) into `shang:/data/esteban/faceight/verdict_{b,c}/
{import.jsonl,README.md}` and uploaded with a copy of the round-1 uploader to `oss://we-vlm-annotation-data-sh/
faceight_{b,c}/{images/,prelabels/}`. Records have the round-1 / faceback-45 importer shape (session, chunk "%03d",
view lview / rview, frame_idx, width 2328, height 1748, boxes as normalized xywh 4 dp clipped to [0, 1], image
`<session>_c<chunk>_f<frame_idx:06d>.jpg`); the eye is the `view` field only, as in faceback-45's import_right.jsonl,
and the names are unique within each round and disjoint from faceight_a and from each other. Prelabel rule: one
detector per frame by cell, armW only cells -> armW boxes, both and AA34 only cells -> armAA34 boxes, faceless ->
none; only boxes with score >= the cell's band edge (0.6+ -> 0.6, [0.3,0.6) -> 0.3, [0.1,0.3) -> 0.1), no second
detector, no sub-edge box, no score. Zero-area boxes after conversion are dropped as in round 1; armW emits point
boxes (x1 == x2, y1 == y2) that count toward sW in the cell rule but cannot be drawn, so 35 (round 2) / 28 (round 3)
frames in the armW-only and AA34-only [0.1,0.3) cells end with 0 boxes. Counts: faceight_b 60,000 frames (28,503
lview / 31,497 rview), 68,980 boxes, 9,635 frames with 0 boxes (9,600 faceless); faceight_c 40,000 (18,994 /
21,006), 47,431 boxes, 6,428 with 0 boxes (6,400 faceless). import.jsonl md5 59b528a05778593f038ad8f94da8feb6 (b),
77392d0f141244930a09e3c4487b1fd2 (c); per-cell tables and source md5s in each package's README.md.

# faceight master split (WOR-1212)

Rule. The faceight draw is split once, at the episode level: every one of the 34,667 episodes in
`data/faceight/episodes.txt` (`faceight == 1` in `data/episode_usage.csv`, 34,667 distinct sessions) gets a
side, frozen forever. An annotation round is a projection and is never drawn on its own: its frames follow
their episode's side, both eyes together. Master lists `data/splits/faceight_train_episodes_v1.txt` (27,734
ids, md5 500bf73f0608447cb28e36707b9a5453) and `data/splits/faceight_eval_episodes_v1.txt` (6,933, md5
03f52f10684232eadca77f870bd54ae2), one id per line, sorted. Script `mining/faceight_master_split.py`, sibling
of `faceight_split.py` (`--dry-run`, `--write-csv`, `--check`, `--episodes/--frames/--csv/--manifest-dir/
--splits-dir/--out-dir` overrides, exit 1 on any failed check).

Pinned, reproduced and never redrawn:
- the 19,783 faceight_a episodes keep the side of `faceight_a_{train,eval}_episodes_v1.txt` (models are
  already trained on that train part). The script asserts that the round-1 projection reproduces both lists
  byte for byte (md5 edb30b18a6b3acae662807750205d5b8 train, 2d2b6b7da58351358a4aa10560160a31 eval, both
  unchanged by this work).
- the 42 drawn episodes whose session is in faceback_45 keep the faceback side (34 are also faceight_a, 8 are
  not). Two-way checks against `episode_usage.csv` and every list in `data/splits/`, as `faceight_split.py`
  does: an episode in a frozen list must carry that dataset and role in the csv, an episode carrying a
  dataset must be in its list, every other drawn episode must be dataset '' / role unused and in no list, and
  a list the script does not know that holds a drawn episode or session is a hard failure.
- pinned 19,791 episodes (3,959 eval, 15,832 train); free 14,876.

Free episodes. Seeded stratified draw with `random.Random(19760703)` (the project seed), strata
scene_category x face-free x face band 0 / 1-2 / 3-6 / 7-15 / 16+ over the episodes that have frames, plus one
band of its own for the 3,653 episodes with `chunk_sel -1` and no frame at all; then the `faceight_split.py`
hill-climb (single moves, then 20,000 sampled pair swaps per sweep; 20 restarts, best kept). Frames means all
744,312 rows of `data/faceight/frames.csv` (rounds 0 to 3, both eyes); face-bearing means an armW face on
round 0 (`n_faces`) and at least one human box on rounds 1 to 3 (the
`training/manifests/faceight_{a,b,c}_labelv2.txt` manifests, joined on the frame name). Objective as in the
faceight_a split: 100 x max eval-share deviation over episodes / frames / face-bearing frames, plus 50 x the
largest per-scene eval episode share deviation (zero within one episode of the target), plus soft terms (face
mass share, face-bearing rate gap, per-scene frame share gaps). Checks: hard axes within 1 pt of 20%, every
scene within 2 pt or one episode.

Achieved (seed 19760703, best restart 0, 6,128,687 candidates scored, max deviation 0.00 pt on every hard axis
and on the per-scene axis). "Faces" is armW faces on round 0 and human boxes on rounds 1 to 3:

| Slice | Episodes train / eval | Eval ep | Frames train / eval | Eval fr | Face-bearing train / eval | Eval fb | Faces train / eval | Eval faces |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| master (rounds 0 to 3) | 27,734 / 6,933 | 20.0% | 595,440 / 148,872 | 20.0% | 99,392 / 24,849 | 20.0% | 225,170 / 56,293 | 20.0% |
| round 0 pool (armW) | 24,811 / 6,203 | 20.0% | 481,321 / 120,583 | 20.0% | 48,904 / 12,291 | 20.1% | 96,543 / 23,897 | 19.8% |
| faceight_a (round 1) | 15,826 / 3,957 | 20.0% | 33,926 / 8,482 | 20.0% | 18,096 / 4,540 | 20.1% | 54,151 / 13,563 | 20.0% |
| faceight_b (round 2) | 20,090 / 5,028 | 20.0% | 48,073 / 11,927 | 19.9% | 19,599 / 4,915 | 20.0% | 51,894 / 13,373 | 20.5% |
| faceight_c (round 3) | 15,008 / 3,696 | 19.8% | 32,120 / 7,880 | 19.7% | 12,793 / 3,103 | 19.5% | 22,582 / 5,460 | 19.5% |
| faceight_b + faceight_c | 23,003 / 5,750 | 20.0% | 80,193 / 19,807 | 19.8% | 32,392 / 8,018 | 19.8% | 74,476 / 18,833 | 20.2% |

Only the overall split is optimised; the per-round shares are whatever the projection gives. faceight_b lands
on 20.0% of episodes (19.9% frames, 20.0% face-bearing frames, 20.5% human boxes). faceight_c sits
consistently just under a fifth: 19.8% episodes, 19.7% frames, 19.5% face-bearing frames, 19.5% human boxes;
b and c together give 20.0% / 19.8% / 19.8% / 20.2%. Nothing drifts by more than 0.5 pt, so no round is
corrected, but faceight_c alone is not exactly 80/20 and a paper quoting it should say 19.8/80.2.

Per scene (episode counts and eval shares are over the whole draw; the last four columns are the eval episode
share inside each round's projection):

| Scene | Episodes | Train ep | Eval ep | Eval ep share | Eval frame share | Eval ep share faceight_a | Eval ep share faceight_b | Eval ep share faceight_c | Eval ep share b + c |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Factory | 16,140 | 12,913 | 3,227 | 20.0% | 20.0% | 20.0% | 20.1% | 19.7% | 20.1% |
| Home | 7,437 | 5,950 | 1,487 | 20.0% | 20.0% | 20.0% | 19.9% | 20.1% | 20.0% |
| Workbench | 3,611 | 2,889 | 722 | 20.0% | 20.0% | 20.0% | 19.1% | 20.1% | 19.4% |
| Supermarket | 2,640 | 2,111 | 529 | 20.0% | 20.0% | 20.0% | 20.1% | 19.3% | 19.9% |
| Store | 1,628 | 1,302 | 326 | 20.0% | 20.0% | 19.9% | 20.2% | 19.3% | 20.1% |
| CraftStudio | 970 | 776 | 194 | 20.0% | 20.0% | 20.0% | 20.5% | 19.6% | 19.7% |
| Logistics | 677 | 541 | 136 | 20.1% | 20.1% | 19.9% | 20.6% | 19.0% | 19.8% |
| Other | 603 | 483 | 120 | 19.9% | 20.0% | 20.1% | 20.8% | 19.8% | 20.5% |
| Healthcare | 403 | 322 | 81 | 20.1% | 20.0% | 20.1% | 20.6% | 20.1% | 19.9% |
| Office | 298 | 238 | 60 | 20.1% | 20.1% | 20.2% | 20.2% | 20.7% | 19.9% |
| Restaurant | 260 | 209 | 51 | 19.6% | 19.9% | 19.5% | 20.5% | 19.3% | 20.4% |

Derived lists, written by projection, never drawn: `data/splits/faceight_b_train_episodes_v1.txt` (20,090,
md5 743f3f1723268f808717f64949cd1280), `faceight_b_eval_episodes_v1.txt` (5,028, md5
d1c55ff5b4d79371024397013c53879f), `faceight_c_train_episodes_v1.txt` (15,008, md5
585e8c1a9102131562e32ccc5407ffa0), `faceight_c_eval_episodes_v1.txt` (3,696, md5
eaca92ed42191669e82eea41c0d4cbda). Rounds 2 and 3 share 15,069 episodes; the script asserts that count and
that every shared episode is on the same side in both projections, which closes the leak reported in WOR-1034
and WOR-987 (a split drawn for faceight_b alone would have leaked into faceight_c and back). The 3,212 / 2,347
faceight_a eval episodes inside rounds 2 / 3 are still eval here, by construction.

`data/episode_usage.csv`: md5 b620d8ac6e76f5676dd29b23dc4cedf7 before, d70653615e3b2034aa2785429987add0 after
`--write-csv`. 33,098 of the 34,667 drawn rows changed; 277,339 rows and every column other than `dataset` and
`role` are byte-identical (`diff <(cut -d, -f1,2,5- before) <(cut -d, -f1,2,5- after)` is empty), no row
outside the draw is touched. `role` is now train or eval on all 34,667 drawn episodes (corpus totals 35,443
train, 8,851 eval, 233,044 unused). `dataset` gains `faceight_b` and `faceight_c` in the compound form for the
episodes annotated in those rounds; existing tokens keep their place and order and are never rewritten, so the
values are `faceight_a+faceight_b+faceight_c` (9,695), `faceight_a+faceight_b` (6,290), `faceight_b+faceight_c`
(5,349), `faceight` (4,345), `faceight_b` (3,744), `faceight_a+faceight_c` (2,196), `faceight_a` (1,568),
`faceight_c` (1,438), and the 42 faceback_45 compounds. Decision: a free episode with no round 1 to 3 frame
(round-0 pool frames only, or no frame at all) gets the plain token `faceight` with role train or eval, 4,345
rows (3,464 train, 881 eval); it records that the episode is drawn and has a frozen side but no labelled round
yet, so a later round can append its token without inventing a side. The 1,569 unchanged rows are faceight_a
(one of them faceback_45+faceight_a) episodes with no round-2 and no round-3 frame: their dataset and role were
already what this split wants.

Deterministic and checked. Two runs give byte-identical lists (md5s above), including a run made after the csv
was written, and `--check` regenerates the split and compares it with the six lists on disk and with the csv.
Sabotage runs, each exit 1 with nothing written: a master eval episode moved to train in a copy of the lists
gives "differs from the regenerated split (1 missing)"; a pinned faceback_45 role flipped in a csv copy gives
"csv role eval != faceback_45 side train"; an extra unknown list in a splits copy holding one drawn episode
gives "unknown list holds 1 drawn faceight episodes/sessions"; a frames.csv copy with one row's session_id
altered gives "episode ... spans 2 sessions"; a `--write-csv` rerun gives "dataset already contains faceight_b;
refusing to re-append" and leaves the csv copy's md5 unchanged; an episode with no round-1 frame added to a
copy of `faceight_a_train_episodes_v1.txt` (and to its csv dataset) gives "the round-1 projection does not
reproduce it (1 extra)".
