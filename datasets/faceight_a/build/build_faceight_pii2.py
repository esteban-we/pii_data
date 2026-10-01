#!/usr/bin/env python3
"""Build the faceight datasets in the pii2 layout (PII-1315, PII-1360): faceight_a (round 1),
faceight_b (round 2), faceight_c (round 3) under /data/esteban/pii/datasets/. Stdlib only,
idempotent; every source is read-only (local pii, shang, OSS: GetObject / HEAD / list only).

    stage-shang --set S    scp the shang-only prelabel files into SHANG_STAGE (read-only on shang),
                           check their md5 against `ssh shang md5sum`; a rerun copies nothing
    pull        --set b|c  GetObject every image of the round from OSS into images/, named
                           <session>_c<chunk>_<eye>_f<frame_idx:06d>.jpg (eye from frames.csv);
                           size == Content-Length == listing size, md5 == ETag; resumable (a file
                           already present with the listed size is skipped); throughput printed
    build       --set S    set a: copy + rename the local images; b, c: require `pull` done.
                           Then frames.csv (size, md5 computed; asserted == OSS ETag), split.csv,
                           boxes/v1 (+ v2 for b, c) with job/prelabels and job/output byte copies
    verify      --set S    md5 every image against frames.csv (and against the local source and
                           the OSS listing where available), row counts, reviewed sets, box files
                           re-derived from the job copies, pinned md5s; exit 1 on any mismatch

Layout written (README.md files are written by hand, not here):

    images/                 byte copies (no links), one file per frames.csv row
    frames.csv              image,session,chunk,eye,frame_idx,t_ms,size,md5,round,local_name,oss_key
                            local_name = the faceight frame file (<session>_c<chunk>_<lview|rview>_t<ms>.jpg,
                            the `file` column of data/faceight/frames.csv); src_oss_key = the eye-less
                            Verdict object on we-vlm-annotation-data-sh
                            (<prefix>/<session>_c<chunk>_f<frame_idx:06d>.jpg) the image was pulled from;
                            oss_key = pii/data/<set>/<image> on algorithm-datasets (PII-1413)
    split.csv               session,role (train|eval), the master faceight split (PII-1212) projected
                            through data/splits/faceight_<set>_{train,eval}_episodes_v1.txt
    boxes/vN/frames.csv     image,reviewed
    boxes/vN/boxes.csv      image,x1,y1,x2,y2,ignore  (pixel xyxy, 1 decimal, sorted by image,x1,y1)
    boxes/vN/job/prelabels/ what the vendor started from (byte copies)
    boxes/vN/job/output/    what the vendor returned (byte copies)

Mapping. Every vendor file names a frame <session>_c<chunk>_f<frame_idx:06d>.jpg with the eye only
in its `view` field; the owner image is looked up on (session, chunk, view, frame_idx) in the rows
of data/faceight/frames.csv at annotation_round == the set's round. Boxes are normalized top-left
xywh and become pixel xyxy as x*W, y*H, (x+w)*W, (y+h)*H at one decimal (W x H = 2328 x 1748,
the faceback_45 convention).

Versions.
  a v1  the round-1 human pass (2026-09-11): output faceight_a/labels/face_boxes_faceight.jsonl,
        prelabels shang:/data/esteban/faceight/verdict/ (import_left.jsonl, prelabels_full.jsonl,
        README.md). reviewed = all. Cross-checked against faceight_a/boxes.jsonl while it exists.
  b v1  the round-2 human pass (2026-09-16): output /data/esteban/tmp/faceight_b_boxes/
        face_boxes_faceight_b.jsonl (+ the labels README from shang), prelabels shang verdict_b/.
  b v2  the armAE34-addition review (PII-1047, 2026-09-18): /data/esteban/tmp/pii/faceight_v2/
        face_boxes_faceight_b_v2.jsonl over v1 on its 9,402 frames (= the frames of
        faceight_b_relabel_v1/import_{left,right}.jsonl that carry an armAE34 box). reviewed = those.
  c v1  the round-3 human pass (2026-09-16): output faceight_c/labels/face_boxes_faceight_c.jsonl,
        prelabels shang verdict_c/ (a local copy of import.jsonl is asserted equal).
  c v2  as b v2 with face_boxes_faceight_c_v2.jsonl (9,925 frames) and faceight_c_relabel_v1.
  b v3  the SECOND review pass over the SAME 9,402-frame package (WOR-1425, drop 2026-09-22):
        /data/esteban/tmp/pii_faceightr_v3/face_boxes_faceight_b_v3.jsonl merged over v1 on those
        frames, exactly as training/merge_faceight_v3.py merged it into the v3 labelv2 manifest.
        The vendor re-reviewed the v2 Verdict dataset itself (drop ds faceight_b_v2, review_round 2,
        machine_src faceight_b_relabel_v1@1), so the prelabel package is v2's byte for byte:
        `prelabel_version` reads it out of boxes/v2/job/prelabels/ and boxes/v3/job/ holds the
        return only. The reviewed frame set is the same as v2's, so v1 and v2 agree off it and
        merging over v1 changes nothing outside the 9,402.
  c v3  as b v3 with face_boxes_faceight_c_v3.jsonl (9,925 frames).
"""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool

# PII-1449: the live store; data/pii_root.py resolves it (env PII_ROOT or PII2_ROOT,
# default /data/esteban/pii). A clone elsewhere (shang, fluence) sets the env var.
# datasets/<name>/build/<this file> is three levels under the store root, so four
# dirname() calls (PII-1681: the PII-1649 move left three, which resolved to
# <store>/datasets and made every builder fail on `import pii_root`).
STORE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(STORE, "data"))
from pii_root import CODE_ROOT, PII_ROOT, open_index  # noqa: E402

# Provenance (PII-1682): the pre-PII-1315 tree this script read. The project retired it, so the
# paths below record where the data came from; they are not a tree to read today.
LEGACY_ROOT = "/data/esteban/pii_backup"

PII = LEGACY_ROOT    # PII-1448: the old tree, now /data/esteban/pii_backup
TMP = "/data/esteban/tmp"
PII2_ROOT = PII_ROOT          # kept: the name every caller and doc already uses
WRITE_ROOT = PII2_ROOT + "/"
SHANG_STAGE = f"{TMP}/pii2_faceight_shang"          # local copies of shang-only files (stage-shang)
V3_DIR = f"{TMP}/pii_faceightr_v3"                  # the WOR-1425 v3 re-review drops
SHANG_DIRS = {"verdict": "/data/esteban/faceight/verdict", "verdict_b": "/data/esteban/faceight/verdict_b",
              "verdict_c": "/data/esteban/faceight/verdict_c",
              "faceight_b_relabel_v1": f"{PII}/datasets/faceight_b_relabel_v1",
              "faceight_c_relabel_v1": f"{PII}/datasets/faceight_c_relabel_v1",
              "faceight_b_labels": f"{PII}/datasets/faceight_b/labels"}   # staged sub-dir -> dir on shang
STATE = os.path.join(CODE_ROOT, ".knuth", "tmp", "faceight_pii2")   # pull logs, nothing the dataset needs
FRAMES_CSV = os.path.join(STORE, "data", "faceight", "frames.csv")
SPLITS = os.path.join(STORE, "data", "splits")
OSS_BUCKET = "we-vlm-annotation-data-sh"

W, H = 2328, 1748
MD5_HEX = re.compile(r"^[0-9a-f]{32}$")
ONE_DECIMAL = re.compile(r"^-?\d+\.\d$")
FRAME_COLS = ["image", "session", "chunk", "eye", "frame_idx", "t_ms", "size", "md5", "round", "local_name",
              "src_oss_key", "oss_key"]
BOX_COLS = ["image", "x1", "y1", "x2", "y2", "ignore"]
EYE_OF_VST = {"vst_left": "left", "vst_right": "right"}
EYE_OF_VIEW = {"lview": "left", "rview": "right"}
VIEW_OF_EYE = {"left": "lview", "right": "rview"}

# md5 of the shang-only files as staged 2026-09-22 (ssh shang md5sum); the job copies are pinned to them
SHANG_MD5 = {
    "verdict/import_left.jsonl": "0462b3f322c723315884f3a5a0938290",
    "verdict/prelabels_full.jsonl": "861ced911657249e5c77f4dcdc0c82bb",
    "verdict/README.md": "5343a10780312ea1c1212c74ad1098c2",
    "verdict_b/import.jsonl": "59b528a05778593f038ad8f94da8feb6",
    "verdict_b/README.md": "43c2e5c41fc54b76f6250ce99c1ed9ee",
    "verdict_c/import.jsonl": "77392d0f141244930a09e3c4487b1fd2",
    "verdict_c/README.md": "10da4833ecc9f37da35455d2c8207405",
    "faceight_b_relabel_v1/import_left.jsonl": "dca2c0e7cb30f04fa2556d345d272698",
    "faceight_b_relabel_v1/import_right.jsonl": "2e415eec0b407f45be144fe65a882b2f",
    "faceight_b_relabel_v1/README.md": "0d61ec2837ed1009f531af02e09b04a7",
    "faceight_b_relabel_v1/stats.json": "deeb6add4943d46915f4548070e14943",
    "faceight_b_relabel_v1/dets_left.jsonl": "39cee7e5e82621140c12a35faac8a1c3",
    "faceight_b_relabel_v1/dets_right.jsonl": "8f26b913b6047889c6b1ea472d4179ef",
    "faceight_c_relabel_v1/import_left.jsonl": "e2835a487d74ede4830b739bdbaa9ce7",
    "faceight_c_relabel_v1/import_right.jsonl": "954773c99cf1457b208605174c268d23",
    "faceight_c_relabel_v1/README.md": "38993c551c2d037ae3fad289ba82fb94",
    "faceight_c_relabel_v1/stats.json": "e20cae7e2a2802181942480e2f9ed1a6",
    "faceight_c_relabel_v1/dets_left.jsonl": "64d451693eff42c7fc9ea7d57c723666",
    "faceight_c_relabel_v1/dets_right.jsonl": "f893f292a00f83e351e5bac1211cb89a",
    "faceight_b_labels/face_boxes_faceight_b.jsonl": "2c1726c751a1499a476222bb60efdbfe",
    "faceight_b_labels/README.md": "2e482de362441eed3ff550e3e57dd21f",
}


def shang(rel: str) -> str:
    return os.path.join(SHANG_STAGE, rel)


def shang_src(rel: str) -> str:
    d, b = rel.split("/", 1)
    return f"{SHANG_DIRS[d]}/{b}"


SETS = {
    "a": {
        "dataset": "faceight_a", "round": "1",
        "n_images": 42408, "n_left": 42408, "n_right": 0,
        "n_sessions": 19783, "n_eval_sessions": 3957, "n_eval_frames": 8482,
        "oss_prefix": "faceight/images/", "oss_bytes": 27185183632,
        "local_images": f"{PII}/datasets/faceight_a/images",          # t-named files, copied + renamed
        "human_jsonl": (f"{PII}/datasets/faceight_a/boxes.jsonl", "8d701208b79f10f708eb307c0b1211da"),
        "versions": {
            "v1": {"kind": "full", "expect_boxes": 67714,
                   "output": f"{PII}/datasets/faceight_a/labels/face_boxes_faceight.jsonl",
                   "output_md5": "335cad2b33205b88689ec20ae61a19f9",
                   "prelabel_import": ["verdict/import_left.jsonl"],
                   "job": {"prelabels": [shang("verdict/import_left.jsonl"), shang("verdict/prelabels_full.jsonl"),
                                         shang("verdict/README.md")],
                           "output": [f"{PII}/datasets/faceight_a/labels/face_boxes_faceight.jsonl",
                                      f"{PII}/datasets/faceight_a/labels/README.md"]},
                   "pinned_md5": "6d9e2b42b527c6e12fa4f6b7e0947039"},
        },
    },
    "b": {
        "dataset": "faceight_b", "round": "2",
        "n_images": 60000, "n_left": 28503, "n_right": 31497,
        "n_sessions": 25118, "n_eval_sessions": 5028, "n_eval_frames": 11927,
        "oss_prefix": "faceight_b/images/", "oss_bytes": 36576699475,
        "local_images": None,
        "human_jsonl": (f"{PII}/datasets/faceight_b/boxes.jsonl", "4a4c5f04eff27312cc8c3113613b375f"),
        "versions": {
            "v1": {"kind": "full", "expect_boxes": 65267,
                   "output": f"{TMP}/faceight_b_boxes/face_boxes_faceight_b.jsonl",
                   "output_md5": "2c1726c751a1499a476222bb60efdbfe",
                   "prelabel_import": ["verdict_b/import.jsonl"], "machine_count_equal": True,
                   "job": {"prelabels": [shang("verdict_b/import.jsonl"), shang("verdict_b/README.md")],
                           "output": [f"{TMP}/faceight_b_boxes/face_boxes_faceight_b.jsonl",
                                      shang("faceight_b_labels/README.md")]},
                   "pinned_md5": "37e00ad0bfdbb03ee60c8d5a4a3d470f"},
            "v2": {"kind": "subset", "n_reviewed": 9402, "expect_boxes": None,
                   "output": f"{TMP}/pii/faceight_v2/face_boxes_faceight_b_v2.jsonl",
                   "output_md5": "cce2d5b389c747faf9a4804590e91ff2",
                   "prelabel_import": ["faceight_b_relabel_v1/import_left.jsonl", "faceight_b_relabel_v1/import_right.jsonl"],
                   "job": {"prelabels": [shang(f"faceight_b_relabel_v1/{n}") for n in
                                         ("import_left.jsonl", "import_right.jsonl", "dets_left.jsonl", "dets_right.jsonl", "README.md", "stats.json")],
                           "output": [f"{TMP}/pii/faceight_v2/face_boxes_faceight_b_v2.jsonl"]},
                   "pinned_md5": "f84a50f1aa1ef723cb5b4ccdaa09eee5"},
            "v3": {"kind": "subset", "n_reviewed": 9402, "expect_boxes": 62507, "base": "v1",
                   "output": f"{V3_DIR}/face_boxes_faceight_b_v3.jsonl",
                   "output_md5": "1b95adbf21a71061eaf58c4ee711d77d",
                   "prelabel_import": ["faceight_b_relabel_v1/import_left.jsonl", "faceight_b_relabel_v1/import_right.jsonl"],
                   "prelabel_version": "v2",   # the same package, already copied under boxes/v2
                   "job": {"output": [f"{V3_DIR}/face_boxes_faceight_b_v3.jsonl"]},
                   "pinned_md5": "317ee5e166c7ecb457893806772cde22"},
        },
    },
    "c": {
        "dataset": "faceight_c", "round": "3",
        "n_images": 40000, "n_left": 18994, "n_right": 21006,
        "n_sessions": 18704, "n_eval_sessions": 3696, "n_eval_frames": 7880,
        "oss_prefix": "faceight_c/images/", "oss_bytes": 24583618577,
        "local_images": None,
        "human_jsonl": (f"{PII}/datasets/faceight_c/boxes.jsonl", "c7a00a0bfac46d7b8000aa7b9fe26922"),
        "versions": {
            "v1": {"kind": "full", "expect_boxes": 28042,
                   "output": f"{PII}/datasets/faceight_c/labels/face_boxes_faceight_c.jsonl",
                   "output_md5": "e6aa449eeacf9624d3003dee6a6cde70",
                   "prelabel_import": ["verdict_c/import.jsonl"], "machine_count_equal": True,
                   "prelabel_twin": f"{TMP}/pii/faceight_c/verdict_c_import.jsonl",   # local copy, asserted equal
                   "job": {"prelabels": [shang("verdict_c/import.jsonl"), shang("verdict_c/README.md")],
                           "output": [f"{PII}/datasets/faceight_c/labels/face_boxes_faceight_c.jsonl"]},
                   "pinned_md5": "270dbc6ab7510c2202b73c4834f6d919"},
            "v2": {"kind": "subset", "n_reviewed": 9925, "expect_boxes": None,
                   "output": f"{TMP}/pii/faceight_v2/face_boxes_faceight_c_v2.jsonl",
                   "output_md5": "39da90d3ffde895bebe87a594a4d6041",
                   "prelabel_import": ["faceight_c_relabel_v1/import_left.jsonl", "faceight_c_relabel_v1/import_right.jsonl"],
                   "job": {"prelabels": [shang(f"faceight_c_relabel_v1/{n}") for n in
                                         ("import_left.jsonl", "import_right.jsonl", "dets_left.jsonl", "dets_right.jsonl", "README.md", "stats.json")],
                           "output": [f"{TMP}/pii/faceight_v2/face_boxes_faceight_c_v2.jsonl"]},
                   "pinned_md5": "a97295954ca9651413ac888a613a942e"},
            "v3": {"kind": "subset", "n_reviewed": 9925, "expect_boxes": 33763, "base": "v1",
                   "output": f"{V3_DIR}/face_boxes_faceight_c_v3.jsonl",
                   "output_md5": "6e937bd530553b9cad5aa2175c416c67",
                   "prelabel_import": ["faceight_c_relabel_v1/import_left.jsonl", "faceight_c_relabel_v1/import_right.jsonl"],
                   "prelabel_version": "v2",
                   "job": {"output": [f"{V3_DIR}/face_boxes_faceight_c_v3.jsonl"]},
                   "pinned_md5": "e3268102de95c60449d24389029159e9"},
        },
    },
}
# pinned_md5 / expect_boxes of the derived box files are filled in from the first build (see PII-1360);
# a value of None means "not pinned yet" and build prints the value to pin.

FAILS: list[str] = []
CFG: dict = {}
DST = ""


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def check(ok: bool, msg: str) -> None:
    if not ok:
        die(msg)


def fail(msg: str) -> None:
    """verify: record and keep going."""
    FAILS.append(msg)
    print(f"FAIL: {msg}")


def fmt(v: float) -> str:
    return f"{v:.1f}"


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def dst(*parts: str) -> str:
    p = os.path.join(DST, *parts)
    check(os.path.abspath(p).startswith(WRITE_ROOT), f"refusing to write outside {WRITE_ROOT}: {p}")
    return p


def write_csv(path: str, header: list[str], rows) -> None:
    check(path.startswith(WRITE_ROOT), f"refusing to write outside {WRITE_ROOT}: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)


def read_csv(path: str, header: list[str]) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        hdr = next(r)
        check(hdr == header, f"{path}: header {hdr} != {header}")
        return [row for row in r]


def box_key(row):
    return (row[0], float(row[1]), float(row[2]))


def group_boxes(rows) -> dict[str, list[tuple]]:
    d: dict[str, list[tuple]] = {}
    for r in rows:
        d.setdefault(r[0], []).append(tuple(r[1:]))
    return d


def flatten(by: dict[str, list[tuple]]) -> list[list[str]]:
    rows = [[im, *b] for im, bl in by.items() for b in bl]
    rows.sort(key=box_key)
    return rows


def csv_bytes(header: list[str], rows) -> bytes:
    lines = [",".join(header)] + [",".join(r) for r in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


def pix(b: dict) -> tuple:
    """normalized top-left xywh -> pixel xyxy strings (WOR-140 convention), ignore=0."""
    return (fmt(b["x"] * W), fmt(b["y"] * H), fmt((b["x"] + b["w"]) * W), fmt((b["y"] + b["h"]) * H), "0")


def sort_boxes(bl: list[tuple]) -> list[tuple]:
    return sorted(bl, key=lambda t: (float(t[0]), float(t[1])))


def _md5_one(path: str) -> tuple[str, str]:
    return os.path.basename(path), md5_of(path)


def md5_dir(root: str, names: list[str], jobs: int, label: str) -> dict[str, str]:
    t0 = time.time()
    out = {}
    with Pool(jobs) as pool:
        for i, (n, m) in enumerate(pool.imap_unordered(_md5_one, [os.path.join(root, n) for n in names], chunksize=64), 1):
            out[n] = m
            if i % 20000 == 0:
                print(f"  md5 {label}: {i}/{len(names)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"md5 {label}: {len(names)} files ({time.time() - t0:.0f}s)", flush=True)
    return out


# ---------------------------------------------------------------- OSS (GetObject / HEAD / list only)

def oss_client():
    """The OSS class of data/oss_pii_upload.py pointed at the annotation bucket. Only request('GET'),
    head() and list() are used here; put/copy/delete are never called."""
    sys.path.insert(0, os.path.join(STORE, "data"))
    import oss_pii_upload as ossmod
    ossmod.BUCKET = OSS_BUCKET
    ossmod.HOST = f"{OSS_BUCKET}.oss-cn-shanghai.aliyuncs.com"
    ak, sk = ossmod.load_creds()
    return ossmod, ak, sk


def oss_list_images() -> dict[str, tuple[int, str]]:
    """{key: (size, etag)} of the set's image prefix; every ETag must be a plain md5 (single-part)."""
    ossmod, ak, sk = oss_client()
    t0 = time.time()
    listing = ossmod.OSS(ak, sk).list(CFG["oss_prefix"])
    bad = [k for k, (_, e) in listing.items() if not MD5_HEX.match(e)]
    check(not bad, f"{len(bad)} OSS objects with a multipart ETag, e.g. {bad[:3]}")
    print(f"oss: {len(listing)} objects, {sum(s for s, _ in listing.values())} bytes under {CFG['oss_prefix']} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return listing


# ---------------------------------------------------------------- sources

class Frame:
    __slots__ = ("image", "session", "chunk", "eye", "frame_idx", "t_ms", "local_name", "oss_key", "episode")

    def __init__(self, image, session, chunk, eye, frame_idx, t_ms, local_name, oss_key, episode):
        self.image, self.session, self.chunk, self.eye = image, session, chunk, eye
        self.frame_idx, self.t_ms, self.local_name, self.oss_key, self.episode = frame_idx, t_ms, local_name, oss_key, episode


def load_round() -> list[Frame]:
    """The set's rows of data/faceight/frames.csv (annotation_round == round), sorted by image.
    Dies if an eye-less OSS name would collide (the spec's stop condition)."""
    frames = []
    with open_index(FRAMES_CSV) as f:
        for d in csv.DictReader(f):
            if d["annotation_round"] != CFG["round"]:
                continue
            session, chunk = d["session_id"], f"{int(d['chunk']):03d}"
            eye, fi, t_ms = EYE_OF_VST[d["view"]], int(d["frame_idx"]), int(d["t_ms"])
            check(d["file"] == f"{session}_c{chunk}_{VIEW_OF_EYE[eye]}_t{t_ms}.jpg", f"frames.csv: file {d['file']} != its fields")
            frames.append(Frame(f"{session}_c{chunk}_{eye}_f{fi:06d}.jpg", session, chunk, eye, fi, t_ms, d["file"],
                                f"{CFG['oss_prefix']}{session}_c{chunk}_f{fi:06d}.jpg", d["episode_id"]))
    frames.sort(key=lambda fr: fr.image)
    check(len(frames) == CFG["n_images"], f"round {CFG['round']} has {len(frames)} rows, expected {CFG['n_images']}")
    check(len({fr.image for fr in frames}) == len(frames), "duplicate image names")
    keys = {}
    for fr in frames:
        check(fr.oss_key not in keys, f"eye-less OSS name collides across eyes: {fr.oss_key} "
                                      f"({keys.get(fr.oss_key)} and {fr.image}); stop, see PII-1360")
        keys[fr.oss_key] = fr.image
    n_left = sum(1 for fr in frames if fr.eye == "left")
    check((n_left, len(frames) - n_left) == (CFG["n_left"], CFG["n_right"]), f"eye counts {n_left}/{len(frames) - n_left}")
    ep2s, s2ep = {}, {}
    for fr in frames:
        check(ep2s.setdefault(fr.episode, fr.session) == fr.session, f"episode {fr.episode} spans two sessions")
        check(s2ep.setdefault(fr.session, fr.episode) == fr.episode, f"session {fr.session} spans two episodes")
    check(len(s2ep) == CFG["n_sessions"], f"{len(s2ep)} sessions, expected {CFG['n_sessions']}")
    return frames


def owner_index(frames: list[Frame]) -> dict[tuple, str]:
    """(session, chunk, view, frame_idx) -> image, the key every vendor file is joined on."""
    return {(fr.session, fr.chunk, VIEW_OF_EYE[fr.eye], fr.frame_idx): fr.image for fr in frames}


def load_splits(frames: list[Frame]) -> dict[str, str]:
    """session -> role from the set's episode lists; every listed episode has frames and vice versa."""
    roles_ep = {}
    for role in ("train", "eval"):
        p = os.path.join(SPLITS, f"faceight_{CFG['set']}_{role}_episodes_v1.txt")
        for e in open(p, encoding="utf-8").read().split():
            check(e not in roles_ep, f"episode {e} in both split files")
            roles_ep[e] = role
    ep2s = {fr.episode: fr.session for fr in frames}
    check(set(ep2s) == set(roles_ep), f"split lists {len(roles_ep)} episodes, frames have {len(ep2s)}; "
                                      f"missing {len(set(ep2s) - set(roles_ep))}, extra {len(set(roles_ep) - set(ep2s))}")
    roles = {ep2s[e]: r for e, r in roles_ep.items()}
    check(sum(1 for r in roles.values() if r == "eval") == CFG["n_eval_sessions"], "eval session count")
    return roles


def load_drop(path: str, owners: dict[tuple, str], expect_md5: str | None = None) -> tuple[dict[str, list[tuple]], dict[str, int]]:
    """Vendor drop jsonl (records + {"kind":"end"} trailer) -> ({owner image: sorted pixel boxes},
    {owner: n_boxes_machine}). Every record must map to a frame of this set."""
    if expect_md5:
        check(md5_of(path) == expect_md5, f"{path}: md5 != pinned {expect_md5}")
    out: dict[str, list[tuple]] = {}
    machine: dict[str, int] = {}
    n, trailer = 0, None
    for ln in open(path, encoding="utf-8"):
        d = json.loads(ln)
        if "session" not in d:
            check(trailer is None and d.get("kind") == "end", f"{path}: unexpected non-record line")
            trailer = d
            continue
        n += 1
        key = (d["session"], d["chunk"], d["view"], int(d["frame_idx"]))
        check(key in owners, f"{path}: record {key} is not a frame of this set")
        name = owners[key]
        check(d["image_uri"].rsplit("/", 1)[-1] == f"{d['session']}_c{d['chunk']}_f{int(d['frame_idx']):06d}.jpg",
              f"{path}: image_uri {d['image_uri']} != its fields")
        check(name not in out, f"{path}: {name} appears twice")
        check((d["width"], d["height"]) == (W, H) and d["n_boxes"] == len(d["boxes"]), f"{path}: dims/n_boxes {name}")
        out[name] = sort_boxes([pix(b) for b in d["boxes"]])
        machine[name] = int(d["n_boxes_machine"])
    check(trailer is not None and trailer["frames"] == n, f"{path}: trailer/record count mismatch")
    return out, machine


def load_import(path: str, owners: dict[tuple, str]) -> dict[str, list[dict]]:
    """Verdict import jsonl (prelabels) -> {owner image: raw box dicts}."""
    out: dict[str, list[dict]] = {}
    for ln in open(path, encoding="utf-8"):
        d = json.loads(ln)
        key = (d["session"], d["chunk"], d["view"], int(d["frame_idx"]))
        check(key in owners, f"{path}: record {key} is not a frame of this set")
        name = owners[key]
        check(d["image"] == f"{d['session']}_c{d['chunk']}_f{int(d['frame_idx']):06d}.jpg", f"{path}: image {d['image']}")
        check(name not in out and (d["width"], d["height"]) == (W, H), f"{path}: dup/dims {name}")
        out[name] = d["boxes"]
    return out


def load_human_jsonl(frames: list[Frame]) -> dict[str, list[tuple]] | None:
    """The pii conversion (faceight_<set>/boxes.jsonl, box_src human) -> {image: sorted boxes},
    or None when the file is gone. Records are keyed by the t-named file (local_name)."""
    path, pinned = CFG["human_jsonl"]
    if not os.path.isfile(path):
        return None
    check(md5_of(path) == pinned, f"{path}: md5 != pinned {pinned}")
    by_local = {fr.local_name: fr for fr in frames}
    out: dict[str, list[tuple]] = {}
    for ln in open(path, encoding="utf-8"):
        d = json.loads(ln)
        fn = d["image"].rsplit("/", 1)[-1]
        check(fn in by_local, f"{path}: {fn} is not a frame of this set")
        fr = by_local[fn]
        check((d["session"], d["chunk"], d["frame"], d["eye"], d["view"], d["t_ms"])
              == (fr.session, fr.chunk, fr.frame_idx, fr.eye, VIEW_OF_EYE[fr.eye], fr.t_ms), f"{path}: fields of {fn}")
        check(d["box_src"] == "human" and d["n_boxes"] == len(d["boxes"]) and fr.image not in out, f"{path}: {fn}")
        out[fr.image] = sort_boxes([pix(b) for b in d["boxes"]])
    check(len(out) == len(frames), f"{path}: {len(out)} rows, expected {len(frames)}")
    return out


# ---------------------------------------------------------------- versions

def derive_versions(frames: list[Frame], job_root) -> dict[str, tuple[list, set]]:
    """{version: (box rows, reviewed set)} re-derived from the job copies
    (job_root(version, sub, basename) -> path)."""
    owners = owner_index(frames)
    image_set = {fr.image for fr in frames}
    out: dict[str, tuple[list, set]] = {}
    by_of: dict[str, dict[str, list[tuple]]] = {}   # version -> {image: boxes}, face-free dropped
    prev_by: dict[str, list[tuple]] = {}
    for version, spec in CFG["versions"].items():
        drop, machine = load_drop(job_root(version, "output", os.path.basename(spec["output"])), owners, spec["output_md5"])
        pre: dict[str, list[dict]] = {}
        pre_v = spec.get("prelabel_version", version)   # v3 re-reviewed v2's package (PII-1681)
        for rel in spec["prelabel_import"]:
            part = load_import(job_root(pre_v, "prelabels", os.path.basename(rel)), owners)
            check(not (set(part) & set(pre)), f"{version}: prelabel files overlap")
            pre.update(part)
        if spec["kind"] == "full":
            check(set(drop) == image_set, f"{version}: drop covers {len(drop)} frames, expected all {len(image_set)}")
            check(set(pre) == image_set, f"{version}: prelabels cover {len(pre)} frames, expected all")
            if spec.get("machine_count_equal"):
                bad = [n for n in drop if machine[n] != len(pre[n])]
                check(not bad, f"{version}: n_boxes_machine != prelabel box count on {len(bad)} frames, e.g. {bad[:3]}")
            by, reviewed = drop, image_set
        else:
            check(set(pre) == image_set, f"{version}: relabel prelabels cover {len(pre)} frames, expected all")
            with_addition = {n for n, bl in pre.items() if any(b.get("source") == "armAE34" for b in bl)}
            check(set(drop) == with_addition, f"{version}: drop frames ({len(drop)}) != frames with an armAE34 "
                                              f"addition in the prelabels ({len(with_addition)})")
            check(len(drop) == spec["n_reviewed"], f"{version}: {len(drop)} reviewed frames, expected {spec['n_reviewed']}")
            base = by_of[spec["base"]] if spec.get("base") else prev_by
            by = {k: v for k, v in base.items() if k not in drop}
            by.update(drop)
            reviewed = set(drop)
        rows = flatten(by)
        if spec.get("expect_boxes") is not None:
            check(len(rows) == spec["expect_boxes"], f"{version}: {len(rows)} boxes, expected {spec['expect_boxes']}")
        got = md5_bytes(csv_bytes(BOX_COLS, rows))
        if spec.get("pinned_md5"):
            check(got == spec["pinned_md5"], f"{version}: boxes.csv md5 {got} != pinned {spec['pinned_md5']}")
        else:
            print(f"  {version}: {len(rows)} boxes, boxes.csv md5 {got} (not pinned yet)")
        out[version] = (rows, reviewed)
        prev_by = {k: v for k, v in by.items() if v}
        by_of[version] = prev_by
    return out


# ---------------------------------------------------------------- stage-shang

def stage_shang(args) -> None:
    """scp the shang-only files of this set (read-only there) into SHANG_STAGE and check their md5
    against `ssh shang md5sum` and the pinned values."""
    staged = {os.path.relpath(p, SHANG_STAGE) for v in CFG["versions"].values()
              for sub in v["job"].values() for p in sub if p.startswith(SHANG_STAGE)}
    want = sorted(staged)
    check(staged <= set(SHANG_MD5), f"no pinned md5 for {staged - set(SHANG_MD5)}")
    todo = [rel for rel in want if not (os.path.isfile(shang(rel)) and md5_of(shang(rel)) == SHANG_MD5[rel])]
    print(f"stage-shang {CFG['dataset']}: {len(want)} files, {len(todo)} to copy")
    for rel in todo:
        os.makedirs(os.path.dirname(shang(rel)), exist_ok=True)
        r = subprocess.run(["scp", "-q", "-o", "BatchMode=yes", f"shang:{shang_src(rel)}", shang(rel)])
        check(r.returncode == 0, f"scp shang:{shang_src(rel)} failed")
    remote = subprocess.run(["ssh", "-o", "BatchMode=yes", "shang", "md5sum " + " ".join(shang_src(rel) for rel in want)],
                            capture_output=True, text=True)
    check(remote.returncode == 0, f"ssh shang md5sum failed: {remote.stderr}")
    remote_md5 = [ln.split()[0] for ln in remote.stdout.splitlines()]
    check(len(remote_md5) == len(want), "shang md5sum returned a different number of lines")
    for rel, rm in zip(want, remote_md5):
        check(rm == SHANG_MD5[rel], f"{rel}: shang md5 {rm} != pinned {SHANG_MD5[rel]} (source changed on shang)")
        check(md5_of(shang(rel)) == rm, f"{rel}: staged copy md5 != shang")
    print(f"stage-shang OK: {len(want)} files equal to shang and to the pinned md5s")


# ---------------------------------------------------------------- pull (b, c)

def pull(args) -> None:
    check(CFG["local_images"] is None, f"set {CFG['set']} has local images; pull is for b and c")
    t0 = time.time()
    frames = load_round()
    listing = oss_list_images()
    want = {fr.oss_key: fr for fr in frames}
    check(set(listing) == set(want), f"OSS listing ({len(listing)}) != frames.csv keys ({len(want)}); "
                                     f"extra {sorted(set(listing) - set(want))[:3]}, missing {sorted(set(want) - set(listing))[:3]}")
    check(sum(s for s, _ in listing.values()) == CFG["oss_bytes"], "OSS byte total changed")
    img_dir = dst("images")
    os.makedirs(img_dir, exist_ok=True)
    todo = [fr for fr in frames if not (os.path.isfile(os.path.join(img_dir, fr.image))
                                        and os.path.getsize(os.path.join(img_dir, fr.image)) == listing[fr.oss_key][0])]
    print(f"pull {CFG['dataset']}: {len(frames)} objects, {len(frames) - len(todo)} present, {len(todo)} to fetch, "
          f"{args.jobs} threads", flush=True)
    if not todo:
        print(f"pull done in {time.time() - t0:.0f}s (nothing to do)")
        return
    ossmod, ak, sk = oss_client()
    local = threading.local()

    def fetch_once(fr: Frame) -> int:
        if not hasattr(local, "c"):
            local.c = ossmod.OSS(ak, sk)
        size, etag = listing[fr.oss_key]
        st, h, body = local.c.request("GET", fr.oss_key)
        if st != 200:
            raise RuntimeError(f"GET {fr.oss_key}: HTTP {st}")
        if not (len(body) == size == int(h["Content-Length"])):
            raise RuntimeError(f"{fr.oss_key}: body {len(body)} != Content-Length {h['Content-Length']} / listed {size}")
        md = md5_bytes(body)
        if not (md == etag == h["ETag"].strip('"').lower()):
            raise RuntimeError(f"{fr.oss_key}: md5 {md} != ETag {h['ETag']} / listed {etag}")
        d = os.path.join(img_dir, fr.image)
        tmp = f"{d}.{threading.get_ident()}.part"
        with open(tmp, "wb") as f:
            f.write(body)
        os.replace(tmp, d)
        return len(body)

    def fetch(fr: Frame) -> int:
        """A truncated or corrupt body is a transport error: retry (OSS.request already retries
        5xx, 429 and socket errors internally)."""
        last = None
        for attempt in range(4):
            try:
                return fetch_once(fr)
            except Exception as e:   # noqa: BLE001
                last = e
                local.c = ossmod.OSS(ak, sk)
                time.sleep(min(20, 2 ** attempt))
        raise RuntimeError(f"{fr.oss_key}: 4 attempts failed, last {last}")

    t1 = time.time()
    done, nbytes, failures, last = 0, 0, [], time.time()
    with ThreadPoolExecutor(args.jobs) as ex:
        futs = {ex.submit(fetch, fr): fr for fr in todo}
        for fut in as_completed(futs):
            try:
                nbytes += fut.result()
                done += 1
            except Exception as e:   # noqa: BLE001
                failures.append((futs[fut].oss_key, str(e)))
                print(f"FAIL {futs[fut].oss_key}: {e}", flush=True)
            if time.time() - last > 30:
                el = time.time() - t1
                print(f"  {done}/{len(todo)} fetched, {nbytes / 1e9:.2f} GB, {nbytes / 1e6 / el:.1f} MB/s, "
                      f"{done / el:.1f} obj/s, failed {len(failures)} ({el:.0f}s)", flush=True)
                last = time.time()
    el = time.time() - t1
    print(f"pull {CFG['dataset']}: fetched {done}/{len(todo)}, {nbytes} bytes in {el:.0f}s = {nbytes / 1e6 / el:.1f} MB/s "
          f"({done / el:.1f} obj/s, {args.jobs} threads), failed {len(failures)}; wall {time.time() - t0:.0f}s", flush=True)
    for k, e in failures[:20]:
        print(f"  failed: {k}: {e}")
    if failures:
        sys.exit(1)


# ---------------------------------------------------------------- build

def copy_local_images(frames: list[Frame]) -> None:
    """Set a: copy the t-named source files into images/ under the eye-ful name (never a link)."""
    src_dir = CFG["local_images"]
    names = sorted(os.listdir(src_dir))
    check(names == sorted(fr.local_name for fr in frames), f"source images in {src_dir} != frames.csv round {CFG['round']} files")
    img_dir = dst("images")
    os.makedirs(img_dir, exist_ok=True)
    copied, t0 = 0, time.time()
    for fr in frames:
        s, d = os.path.join(src_dir, fr.local_name), os.path.join(img_dir, fr.image)
        check(not os.path.islink(s), f"source image is a symlink: {s}")
        if os.path.exists(d) and os.path.getsize(d) == os.path.getsize(s):
            continue
        shutil.copy2(s, d)
        copied += 1
        if copied % 5000 == 0:
            print(f"  images: {copied} copied ({time.time() - t0:.0f}s)", flush=True)
    print(f"images: {copied} copied, {len(frames) - copied} already present ({time.time() - t0:.0f}s)", flush=True)


def copy_job_files(version: str) -> None:
    for sub, files in CFG["versions"][version]["job"].items():
        out_dir = dst("boxes", version, "job", sub)
        os.makedirs(out_dir, exist_ok=True)
        for s in files:
            check(os.path.isfile(s), f"job source missing: {s}" + (" (run stage-shang)" if s.startswith(SHANG_STAGE) else ""))
            if s.startswith(SHANG_STAGE):
                rel = os.path.relpath(s, SHANG_STAGE)
                check(md5_of(s) == SHANG_MD5[rel], f"{s}: md5 != pinned shang md5")
            d = os.path.join(out_dir, os.path.basename(s))
            if os.path.exists(d) and md5_of(d) == md5_of(s):
                continue
            shutil.copyfile(s, d)
        print(f"  {version}/job/{sub}: {len(files)} files")


def build(args) -> None:
    t0 = time.time()
    os.makedirs(dst(), exist_ok=True)
    frames = load_round()
    names = [fr.image for fr in frames]
    if CFG["local_images"]:
        copy_local_images(frames)
    img_dir = dst("images")
    missing = [fr.image for fr in frames if not os.path.isfile(os.path.join(img_dir, fr.image))]
    check(not missing, f"{len(missing)} images missing under images/, e.g. {missing[:3]}"
                       + ("" if CFG["local_images"] else f" (run: build_faceight_pii2.py pull --set {CFG['set']})"))
    extra = sorted(set(os.listdir(img_dir)) - set(names))
    check(not extra, f"{len(extra)} unexpected files under images/, e.g. {extra[:3]}")

    # size and md5 of every copied image, asserted equal to the OSS listing (and the local source for a)
    size = {n: os.path.getsize(os.path.join(img_dir, n)) for n in names}
    md5 = md5_dir(img_dir, names, args.jobs, "images")
    if not args.no_oss:
        listing = oss_list_images()
        check(set(listing) == {fr.oss_key for fr in frames}, "OSS listing keys != frames.csv oss_key")
        bad = [fr.image for fr in frames if listing[fr.oss_key] != (size[fr.image], md5[fr.image])]
        check(not bad, f"{len(bad)} images differ from the OSS size/ETag, e.g. {bad[:3]}")
        print(f"images: size and md5 equal to the OSS listing for all {len(names)}")
    if CFG["local_images"]:
        src_md5 = md5_dir(CFG["local_images"], [fr.local_name for fr in frames], args.jobs, "source")
        bad = [fr.image for fr in frames if src_md5[fr.local_name] != md5[fr.image]]
        check(not bad, f"{len(bad)} copied images differ from the source, e.g. {bad[:3]}")
        print(f"images: md5 equal to the source for all {len(names)}")
    check(sum(size.values()) == CFG["oss_bytes"], f"byte total {sum(size.values())} != OSS total {CFG['oss_bytes']}")

    rows = [[fr.image, fr.session, fr.chunk, fr.eye, str(fr.frame_idx), str(fr.t_ms), str(size[fr.image]), md5[fr.image],
             CFG["round"], fr.local_name, fr.oss_key,
             f"pii/data/{CFG['dataset']}/{fr.image}"] for fr in frames]
    roles = load_splits(frames)
    eval_frames = {fr.image for fr in frames if roles[fr.session] == "eval"}
    check(len(eval_frames) == CFG["n_eval_frames"], f"{len(eval_frames)} eval frames, expected {CFG['n_eval_frames']}")
    write_csv(dst("frames.csv"), FRAME_COLS, rows)
    write_csv(dst("split.csv"), ["session", "role"], [[s, roles[s]] for s in sorted(roles)])
    print(f"frames.csv {len(rows)} rows ({sum(size.values())} bytes; {CFG['n_left']} left, {CFG['n_right']} right), "
          f"split.csv {len(roles)} sessions ({CFG['n_eval_sessions']} eval, {len(eval_frames)} eval frames)")

    for version, spec in CFG["versions"].items():
        copy_job_files(version)
        twin = spec.get("prelabel_twin")
        if twin and os.path.isfile(twin):
            check(md5_of(twin) == SHANG_MD5[os.path.relpath(spec["job"]["prelabels"][0], SHANG_STAGE)],
                  f"{twin} differs from the shang import.jsonl")
    versions = derive_versions(frames, lambda v, sub, b: dst("boxes", v, "job", sub, b))
    human = load_human_jsonl(frames)
    if human is not None:
        v1_by = group_boxes(versions["v1"][0])
        check({k: v for k, v in human.items() if v} == v1_by, "v1 boxes differ from the pii conversion boxes.jsonl")
        print(f"v1 boxes equal to {CFG['human_jsonl'][0]} on all {len(human)} frames")
    for version, spec in CFG["versions"].items():
        box_rows, reviewed = versions[version]
        write_csv(dst("boxes", version, "boxes.csv"), BOX_COLS, box_rows)
        write_csv(dst("boxes", version, "frames.csv"), ["image", "reviewed"],
                  [[n, "1" if n in reviewed else "0"] for n in names])
        by = group_boxes(box_rows)
        ev = [r for r in box_rows if r[0] in eval_frames]
        rv = [r for r in box_rows if r[0] in reviewed]
        print(f"boxes/{version}: {len(box_rows)} boxes on {len(by)} images ({len(names) - len(by)} face-free); "
              f"train {len(box_rows) - len(ev)} boxes, eval {len(ev)} boxes; {len(reviewed)} reviewed carrying "
              f"{len(rv)} boxes on {len({r[0] for r in rv})} images; md5 {md5_of(dst('boxes', version, 'boxes.csv'))}")
    print(f"build done in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- verify

def verify(args) -> None:
    t0 = time.time()
    frames_csv = read_csv(dst("frames.csv"), FRAME_COLS)
    images = [r[0] for r in frames_csv]
    image_set = set(images)
    if len(frames_csv) != CFG["n_images"]:
        fail(f"frames.csv has {len(frames_csv)} rows, expected {CFG['n_images']}")
    if images != sorted(images) or len(image_set) != len(images):
        fail("frames.csv not sorted by image or has duplicates")
    for r in frames_csv:
        image, session, chunk, eye, fi, t_ms, size, md5, rnd, local_name, src_oss_key, oss_key = r
        if (image != f"{session}_c{chunk}_{eye}_f{int(fi):06d}.jpg" or eye not in VIEW_OF_EYE
                or local_name != f"{session}_c{chunk}_{VIEW_OF_EYE.get(eye)}_t{t_ms}.jpg"
                or src_oss_key != f"{CFG['oss_prefix']}{session}_c{chunk}_f{int(fi):06d}.jpg"
                or oss_key != f"pii/data/{CFG['dataset']}/{image}"
                or rnd != CFG["round"] or not MD5_HEX.match(md5) or not size.isdigit()):
            fail(f"frames.csv: bad row for {image}")
            break
    size_of = {r[0]: int(r[6]) for r in frames_csv}
    md5_col = {r[0]: r[7] for r in frames_csv}
    if sum(size_of.values()) != CFG["oss_bytes"]:
        fail(f"frames.csv byte total {sum(size_of.values())} != {CFG['oss_bytes']}")
    frames = None
    if os.path.isfile(FRAMES_CSV) or os.path.isfile(FRAMES_CSV + ".gz"):
        frames = load_round()
        if [[fr.image, fr.session, fr.chunk, fr.eye, str(fr.frame_idx), str(fr.t_ms), CFG["round"], fr.local_name, fr.oss_key]
                for fr in frames] != [[r[0], r[1], r[2], r[3], r[4], r[5], r[8], r[9], r[10]] for r in frames_csv]:
            fail("frames.csv differs from data/faceight/frames.csv (mapping columns)")
    else:
        print(f"  note: {FRAMES_CSV} not present, mapping columns not cross-checked")
        frames = [Frame(r[0], r[1], r[2], r[3], int(r[4]), int(r[5]), r[9], r[10], None) for r in frames_csv]

    split = read_csv(dst("split.csv"), ["session", "role"])
    roles = {s: r for s, r in split}
    if len(roles) != CFG["n_sessions"] or len(split) != CFG["n_sessions"]:
        fail(f"split.csv has {len(split)} rows, expected {CFG['n_sessions']}")
    if {r[1] for r in frames_csv} != set(roles) or set(roles.values()) - {"train", "eval"}:
        fail("split.csv sessions do not match frames.csv, or bad role")
    if sum(1 for r in roles.values() if r == "eval") != CFG["n_eval_sessions"]:
        fail("split.csv eval session count")
    if os.path.isdir(SPLITS) and frames[0].episode is not None and load_splits(frames) != roles:
        fail("split.csv differs from the data/splits episode lists")
    eval_frames = {r[0] for r in frames_csv if roles.get(r[1]) == "eval"}
    if len(eval_frames) != CFG["n_eval_frames"]:
        fail(f"{len(eval_frames)} eval frames, expected {CFG['n_eval_frames']}")

    # images: every file present, no extras, no links, size and md5 equal to frames.csv (and source / OSS)
    img_dir = dst("images")
    listed = sorted(os.listdir(img_dir))
    if listed != images:
        fail(f"images/ lists {len(listed)} files; frames.csv has {len(images)}; "
             f"extra {sorted(set(listed) - image_set)[:3]}, missing {sorted(image_set - set(listed))[:3]}")
    for n in listed:
        p = os.path.join(img_dir, n)
        if os.path.islink(p):
            fail(f"images/{n} is a symlink")
        elif n in size_of and os.path.getsize(p) != size_of[n]:
            fail(f"images/{n}: size {os.path.getsize(p)} != frames.csv size {size_of[n]}")
    src_dir = CFG["local_images"]
    have_src = bool(src_dir) and os.path.isdir(src_dir)
    if have_src:
        for fr in frames:
            s, d = os.path.join(src_dir, fr.local_name), os.path.join(img_dir, fr.image)
            if os.path.isfile(s) and os.path.isfile(d):
                ss, ds_ = os.stat(s), os.stat(d)
                if ss.st_ino == ds_.st_ino and ss.st_dev == ds_.st_dev:
                    fail(f"images/{fr.image} is a hardlink of the source")
    listed_set = set(listed)
    present = [n for n in images if n in listed_set]
    if args.no_md5:
        print("images: md5 check skipped (--no-md5)")
    else:
        got = md5_dir(img_dir, present, args.jobs, "images")
        bad = [n for n in present if got[n] != md5_col[n]]
        for n in bad[:20]:
            fail(f"images/{n}: md5 {got[n]} != frames.csv {md5_col[n]}")
        if bad:
            fail(f"{len(bad)} images differ from frames.csv md5")
        else:
            print(f"images: {len(present)} md5 equal to frames.csv ({time.time() - t0:.0f}s)")
        if have_src:
            local_of = {fr.image: fr.local_name for fr in frames}
            src = md5_dir(src_dir, [local_of[n] for n in present], args.jobs, "source")
            bad = [n for n in present if src[local_of[n]] != got[n]]
            if bad:
                fail(f"{len(bad)} images differ from the source, e.g. {bad[:3]}")
            else:
                print(f"images: {len(present)} md5 equal to the source ({time.time() - t0:.0f}s)")
        elif src_dir:
            print(f"  note: source {src_dir} gone, images compared to frames.csv md5 only")
    if args.no_oss:
        print("oss: listing check skipped (--no-oss)")
    else:
        try:
            listing = oss_list_images()
        except Exception as e:   # noqa: BLE001
            listing = None
            print(f"  note: OSS listing unavailable ({e}); size/md5 not compared to the ETags")
        if listing is not None:
            if set(listing) != {r[10] for r in frames_csv}:
                fail("OSS listing keys != frames.csv oss_key column")
            bad = [r[0] for r in frames_csv if listing.get(r[10]) != (int(r[6]), r[7])]
            if bad:
                fail(f"{len(bad)} frames.csv rows differ from the OSS size/ETag, e.g. {bad[:3]}")
            else:
                print(f"oss: size and ETag equal to frames.csv for all {len(frames_csv)} objects")

    # job copies (compared to the source while it exists, and to the pinned shang md5s)
    for version, spec in CFG["versions"].items():
        for sub, files in spec["job"].items():
            for s in files:
                d = dst("boxes", version, "job", sub, os.path.basename(s))
                if not os.path.isfile(d):
                    fail(f"missing job copy {d}")
                    continue
                if s.startswith(SHANG_STAGE):
                    if md5_of(d) != SHANG_MD5[os.path.relpath(s, SHANG_STAGE)]:
                        fail(f"job copy differs from the pinned shang md5: {d}")
                elif os.path.isfile(s):
                    if md5_of(s) != md5_of(d):
                        fail(f"job copy differs from source: {d}")
                else:
                    print(f"  note: source gone, job copy not compared: {d}")
    if any("missing job copy" in f for f in FAILS):
        print(f"VERIFY FAILED: {len(FAILS)} problem(s), job copies missing so box files not re-derived")
        sys.exit(1)

    expected = derive_versions(frames, lambda v, sub, b: dst("boxes", v, "job", sub, b))
    human = load_human_jsonl(frames)
    if human is not None and {k: v for k, v in human.items() if v} != group_boxes(expected["v1"][0]):
        fail("v1 boxes differ from the pii conversion boxes.jsonl")
    for version, spec in CFG["versions"].items():
        exp_rows, exp_reviewed = expected[version]
        vf = read_csv(dst("boxes", version, "frames.csv"), ["image", "reviewed"])
        if [r[0] for r in vf] != images:
            fail(f"boxes/{version}/frames.csv image column != frames.csv")
        if set(r[1] for r in vf) - {"0", "1"}:
            fail(f"boxes/{version}/frames.csv: reviewed not 0/1")
        reviewed = {r[0] for r in vf if r[1] == "1"}
        if reviewed != exp_reviewed:
            fail(f"boxes/{version}/frames.csv reviewed set: {len(reviewed)} frames, expected {len(exp_reviewed)}")
        rows = read_csv(dst("boxes", version, "boxes.csv"), BOX_COLS)
        if spec.get("expect_boxes") is not None and len(rows) != spec["expect_boxes"]:
            fail(f"boxes/{version}/boxes.csv has {len(rows)} rows, expected {spec['expect_boxes']}")
        if [box_key(r) for r in rows] != sorted(box_key(r) for r in rows):
            fail(f"boxes/{version}/boxes.csv not sorted by (image, x1, y1)")
        if {r[0] for r in rows} - image_set:
            fail(f"boxes/{version}/boxes.csv: box on an image not in frames.csv")
        if any(not all(ONE_DECIMAL.match(v) for v in r[1:5]) or r[5] not in ("0", "1") for r in rows):
            fail(f"boxes/{version}/boxes.csv: coordinate not one decimal or bad ignore")
        if rows != exp_rows:
            fail(f"boxes/{version}/boxes.csv differs from the rows re-derived from job/")
        if spec.get("pinned_md5") and md5_of(dst("boxes", version, "boxes.csv")) != spec["pinned_md5"]:
            fail(f"boxes/{version}/boxes.csv md5 != pinned md5")
        if version != "v1":
            base = spec.get("base") or f"v{int(version[1:]) - 1}"
            prev = read_csv(dst("boxes", base, "boxes.csv"), BOX_COLS)
            if [r for r in prev if r[0] not in reviewed] != [r for r in rows if r[0] not in reviewed]:
                fail(f"boxes/{version}: a frame outside the reviewed set changed against {base}")
        print(f"boxes/{version}: {len(rows)} rows, {len({r[0] for r in rows})} images with boxes, "
              f"{len(reviewed)} reviewed ({time.time() - t0:.0f}s)")

    for p in ["README.md"] + [f"boxes/{v}/README.md" for v in CFG["versions"]]:
        if not os.path.isfile(dst(p)):
            fail(f"missing {p}")
        elif "—" in open(dst(p), encoding="utf-8").read():
            fail(f"{p} contains an em-dash")
    for p in ("_download.log", "_launch_time.txt"):
        if os.path.exists(dst(p)):
            fail(f"{p} must not be in the dataset")
    parts = [n for n in os.listdir(img_dir) if n.endswith(".part")]
    if parts:
        fail(f"{len(parts)} .part files under images/")

    if FAILS:
        print(f"VERIFY FAILED: {len(FAILS)} problem(s) ({time.time() - t0:.0f}s)")
        sys.exit(1)
    print(f"VERIFY OK ({time.time() - t0:.0f}s)")


def main() -> None:
    global CFG, DST
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("stage-shang", "pull", "build", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--set", choices=("a", "b", "c"), required=True)
        if name == "pull":
            p.add_argument("--jobs", type=int, default=128, help="parallel GetObject requests")
        if name in ("build", "verify"):
            p.add_argument("--jobs", type=int, default=16, help="md5 workers")
            p.add_argument("--no-oss", action="store_true", help="skip the OSS listing cross-check")
        if name == "verify":
            p.add_argument("--no-md5", action="store_true", help="skip the per-image md5 pass")
    args = ap.parse_args()
    CFG = dict(SETS[args.set], set=args.set)
    DST = os.path.join(WRITE_ROOT, "datasets", CFG["dataset"])
    os.makedirs(STATE, exist_ok=True)
    {"stage-shang": stage_shang, "pull": pull, "build": build, "verify": verify}[args.cmd](args)


if __name__ == "__main__":
    main()
