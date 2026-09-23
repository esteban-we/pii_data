#!/usr/bin/env python
"""Build the verifier probe set: crop boxes (+20% margin) from face_mine_v1.

Buckets:
  high   ~50 crops, score >= 0.8      (expected obvious faces)
  low    ~50 crops, score in lowest band (expected mostly junk)
  mid    ~20 crops, score ~0.45-0.55  (borderline)

Deterministic (seed 34). Writes crops + probe_manifest.json.
"""
import json, os, random
from PIL import Image

BOXES = '/home/esteban/repos/pii/.knuth/pages/media/face-mine/boxes.json'
IMG_DIR = '/data/esteban/pii/datasets/face_mine_v1/images'
OUT = '/data/esteban/pii/runs/verifier_probe/crops'
MARGIN = 0.20

d = json.load(open(BOXES))
entries = []  # (file, box_idx, x1,y1,x2,y2, score)
for im in d['images']:
    for i, b in enumerate(im['boxes']):
        entries.append((im['file'], i, *b))

rng = random.Random(34)
high = [e for e in entries if e[6] >= 0.8]
low = sorted(entries, key=lambda e: e[6])[:3000]  # lowest 3000 by score
mid = [e for e in entries if 0.45 <= e[6] <= 0.55]
picks = [('high', e) for e in rng.sample(high, 50)] + \
        [('low', e) for e in rng.sample(low, 50)] + \
        [('mid', e) for e in rng.sample(mid, 20)]

manifest = []
for n, (bucket, (f, bi, x1, y1, x2, y2, score)) in enumerate(picks):
    img = Image.open(os.path.join(IMG_DIR, f))
    W, H = img.size
    # boxes are in 2328x1748 space; rescale if image differs
    sx, sy = W / 2328.0, H / 1748.0
    bx1, by1, bx2, by2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
    w, h = bx2 - bx1, by2 - by1
    mx, my = w * MARGIN, h * MARGIN
    cx1, cy1 = max(0, int(bx1 - mx)), max(0, int(by1 - my))
    cx2, cy2 = min(W, int(bx2 + mx)), min(H, int(by2 + my))
    crop = img.crop((cx1, cy1, cx2, cy2))
    name = f'{n:03d}_{bucket}_s{score:.2f}.jpg'
    crop.save(os.path.join(OUT, name), quality=95)
    manifest.append(dict(crop=name, bucket=bucket, file=f, box_idx=bi,
                         score=score, box=[x1, y1, x2, y2],
                         crop_px=[cx1, cy1, cx2, cy2], img_size=[W, H]))

json.dump(manifest, open('/data/esteban/pii/runs/verifier_probe/probe_manifest.json', 'w'), indent=1)
print('wrote', len(manifest), 'crops')
sizes = [(m['crop_px'][2]-m['crop_px'][0], m['crop_px'][3]-m['crop_px'][1]) for m in manifest]
print('min crop', min(sizes), 'max crop', max(sizes))
