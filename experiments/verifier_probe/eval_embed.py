#!/usr/bin/env python
"""SigLIP2 zero-shot + SigLIP2/DINOv2 linear probe (5-fold CV) on the probe crops.

Usage: CUDA_VISIBLE_DEVICES=3 python eval_embed.py
"""
import json, time
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoImageProcessor

W = '/data/esteban/pii/runs/verifier_probe'
man = json.load(open(f'{W}/probe_manifest.json'))
imgs = [Image.open(f"{W}/crops/{m['crop']}").convert('RGB') for m in man]
dev = 'cuda'

out = {}

# ---- SigLIP2 zero-shot ----
sig_id = 'google/siglip2-so400m-patch16-384'
model = AutoModel.from_pretrained(sig_id, torch_dtype=torch.float16).to(dev).eval()
proc = AutoProcessor.from_pretrained(sig_id)
pos = ["a photo of a human face", "a close-up photo of a person's face",
       "a blurry photo of a human face"]
neg = ["the back of a person's head", "a photo of hair", "a photo of an ear",
       "a photo of an object", "a photo of clothing", "a photo of a hand",
       "a photo of a room", "a stuffed animal toy", "a blurry photo of an arm"]
texts = pos + neg
with torch.no_grad():
    ti = proc(text=texts, padding='max_length', return_tensors='pt').to(dev)
    def _feat(x):
        return x if torch.is_tensor(x) else x.pooler_output
    tfeat = _feat(model.get_text_features(**ti))
    tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
    feats, t0 = [], time.time()
    for i in range(0, len(imgs), 32):
        ii = proc(images=imgs[i:i+32], return_tensors='pt').to(dev)
        f = _feat(model.get_image_features(pixel_values=ii['pixel_values'].half()))
        feats.append(f / f.norm(dim=-1, keepdim=True))
    ifeat = torch.cat(feats)
    dt = time.time() - t0
    logits = (ifeat @ tfeat.T).float() * model.logit_scale.exp().float() + model.logit_bias.float()
    probs = torch.softmax(logits, dim=-1)
    p_face = probs[:, :len(pos)].sum(-1).cpu().numpy()
out['siglip_zeroshot'] = dict(crops_per_s=round(len(imgs)/dt, 2),
    results=[dict(crop=m['crop'], label=m['label'], p_face=float(p),
                  pred='face' if p >= 0.5 else 'noface') for m, p in zip(man, p_face)])
sig_emb = ifeat.float().cpu().numpy()
del model
torch.cuda.empty_cache()

# ---- DINOv2 embeddings ----
din_id = 'facebook/dinov2-base'
dmodel = AutoModel.from_pretrained(din_id, torch_dtype=torch.float16).to(dev).eval()
dproc = AutoImageProcessor.from_pretrained(din_id)
with torch.no_grad():
    feats, t0 = [], time.time()
    for i in range(0, len(imgs), 32):
        ii = dproc(images=imgs[i:i+32], return_tensors='pt').to(dev)
        o = dmodel(pixel_values=ii['pixel_values'].half())
        feats.append(o.pooler_output.float().cpu())
    din_emb = torch.cat(feats).numpy()
    dt = time.time() - t0
out['dinov2_speed_crops_per_s'] = round(len(imgs)/dt, 2)

# ---- 5-fold CV linear probes on firm labels ----
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
y = np.array([1 if m['label'] == 'face' else 0 for m in man])
firm = np.array([m['label'] in ('face', 'noface') for m in man])
for name, X in (('siglip_probe', sig_emb), ('dinov2_probe', din_emb)):
    Xf, yf = X[firm], y[firm]
    skf = StratifiedKFold(5, shuffle=True, random_state=34)
    preds = np.zeros(len(yf))
    for tr, te in skf.split(Xf, yf):
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xf[tr], yf[tr])
        preds[te] = clf.predict_proba(Xf[te])[:, 1]
    tfa = ((preds >= .5) & (yf == 1)).sum() / (yf == 1).sum()
    jr = ((preds < .5) & (yf == 0)).sum() / (yf == 0).sum()
    out[name] = dict(cv_true_face_accept=round(float(tfa), 3),
                     cv_junk_reject=round(float(jr), 3),
                     n=int(firm.sum()))
    print(name, out[name])

json.dump(out, open(f'{W}/out_embed.json', 'w'), indent=1)
zs = out['siglip_zeroshot']['results']
firm_zs = [r for r in zs if r['label'] != 'uncertain']
tfa = sum(1 for r in firm_zs if r['label'] == 'face' and r['pred'] == 'face') / sum(1 for r in firm_zs if r['label'] == 'face')
jr = sum(1 for r in firm_zs if r['label'] == 'noface' and r['pred'] == 'noface') / sum(1 for r in firm_zs if r['label'] == 'noface')
print('siglip zeroshot: tfa', round(tfa, 3), 'junk_rej', round(jr, 3), 'speed', out['siglip_zeroshot']['crops_per_s'])
