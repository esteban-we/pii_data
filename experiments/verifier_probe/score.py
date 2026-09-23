#!/usr/bin/env python
"""Score a model's probe run against my eyeballed labels.

Usage: python score.py out_qwen3_8b.json [out_b.json ...]
Headline metrics use only face/noface labels; uncertain crops reported separately.
"""
import json, sys

runs = [json.load(open(p)) for p in sys.argv[1:]]
for r in runs:
    res = r['results']
    labeled = [x for x in res if x['label'] in ('face', 'noface')]
    tp = sum(1 for x in labeled if x['label'] == 'face' and x['pred'] == 'face')
    fn = sum(1 for x in labeled if x['label'] == 'face' and x['pred'] != 'face')
    tn = sum(1 for x in labeled if x['label'] == 'noface' and x['pred'] == 'noface')
    fp = sum(1 for x in labeled if x['label'] == 'noface' and x['pred'] != 'noface')
    print(f"\n== {r['model']}  ({r['crops_per_s']} crops/s, load {r['load_s']}s)")
    print(f"  true-face acceptance : {tp}/{tp+fn} = {tp/(tp+fn):.3f}")
    print(f"  junk rejection       : {tn}/{tn+fp} = {tn/(tn+fp):.3f}")
    print(f"  accuracy (104 firm)  : {(tp+tn)/len(labeled):.3f}")
    print(f"  errors:")
    for x in labeled:
        if (x['label'] == 'face') != (x['pred'] == 'face'):
            print(f"    {x['crop']}: label={x['label']} pred={x['pred']} p_face={x['p_face']}")
    unc = [x for x in res if x['label'] == 'uncertain']
    uf = sum(1 for x in unc if x['pred'] == 'face')
    print(f"  uncertain crops voted face: {uf}/{len(unc)}")

if len(runs) > 1:
    print('\n== disagreements between models (firm labels) ==')
    by = [{x['crop']: x for x in r['results']} for r in runs]
    for crop in by[0]:
        preds = [b[crop]['pred'] for b in by]
        if len(set(preds)) > 1:
            lab = by[0][crop]['label']
            print(f"  {crop} label={lab}: " + ' | '.join(f"{r['model'].split('/')[-1]}={p}" for r, p in zip(runs, preds)))
