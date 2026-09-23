#!/usr/bin/env python
"""Threshold sweep over p_face for a probe run: TFA / JR at each cutoff."""
import json, sys

r = json.load(open(sys.argv[1]))
res = [x for x in r['results'] if x['label'] in ('face', 'noface')]
missing = [x for x in res if x['p_face'] is None]
if missing:
    print(f'warning: {len(missing)} crops lack p_face, treated as pred-based')
print(f"model {r['model']}")
print('thr   TFA     JR      face_miss junk_leak')
for thr in (0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.75, 0.9):
    def acc(x):
        return (x['p_face'] >= thr) if x['p_face'] is not None else (x['pred'] == 'face')
    f = [x for x in res if x['label'] == 'face']
    j = [x for x in res if x['label'] == 'noface']
    tfa = sum(map(acc, f)) / len(f)
    jr = sum(1 for x in j if not acc(x)) / len(j)
    leak = [x['crop'][:3] for x in j if acc(x)]
    miss = [x['crop'][:3] for x in f if not acc(x)]
    print(f'{thr:.2f}  {tfa:.3f}  {jr:.3f}  n={len(miss):2d} {" ".join(miss[:8])}{"..." if len(miss)>8 else ""}  leak={" ".join(leak)}')
