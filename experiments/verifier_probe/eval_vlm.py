#!/usr/bin/env python
"""Run a VLM on the probe crops with vLLM offline inference.

Usage: CUDA_VISIBLE_DEVICES=3 python eval_vlm.py <model_id> <out_json> [max_model_len]
Asks FACE/NOFACE, extracts P(FACE) from first-token logprobs.
"""
import json, os, sys, time, math

MODEL = sys.argv[1]
OUT = sys.argv[2]
MAXLEN = int(sys.argv[3]) if len(sys.argv) > 3 else 4096

from vllm import LLM, SamplingParams
from PIL import Image

W = '/data/esteban/pii/runs/verifier_probe'
man = json.load(open(f'{W}/probe_manifest.json'))

PROMPT = (
    "This is a small crop from a video frame, boxed by an annotator who thought it might be a human face. "
    "Answer FACE only if a real human face is visible with at least part of the facial-feature region "
    "(eyes, nose, or mouth area) discernible; a masked face with a visible eye region counts. "
    "Answer NOFACE for: back of head, top of head, hair only, an ear only, other body parts, "
    "objects, animals, or faces of dolls/toys/posters/screens. "
    "Reply with exactly one word: FACE or NOFACE."
)

t0 = time.time()
llm = LLM(model=MODEL, max_model_len=MAXLEN, gpu_memory_utilization=0.90,
          limit_mm_per_prompt={'image': 1}, enforce_eager=False)
t_load = time.time() - t0

msgs = []
for m in man:
    img = Image.open(f"{W}/crops/{m['crop']}").convert('RGB')
    # upscale tiny crops so the vision tower gets enough patches
    if max(img.size) < 112:
        f = 112 / max(img.size)
        img = img.resize((max(1,int(img.width*f)), max(1,int(img.height*f))), Image.LANCZOS)
    msgs.append([{'role': 'user', 'content': [
        {'type': 'image_pil', 'image_pil': img},
        {'type': 'text', 'text': PROMPT}]}])

sp = SamplingParams(temperature=0, max_tokens=8, logprobs=20)
t1 = time.time()
outs = llm.chat(msgs, sp)
t_inf = time.time() - t1

res = []
for m, o in zip(man, outs):
    text = o.outputs[0].text.strip()
    p_face = None
    # first-token logprobs: aggregate prob mass on FACE-ish vs NOFACE-ish tokens
    try:
        lps = o.outputs[0].logprobs[0]
        pf = pn = 0.0
        for tid, lp in lps.items():
            tok = (lp.decoded_token or '').strip().upper()
            p = math.exp(lp.logprob)
            if tok in ('FACE', 'FA', 'F'): pf += p
            elif tok in ('NOFACE', 'NO', 'N', 'NOF'): pn += p
        if pf + pn > 0: p_face = pf / (pf + pn)
    except Exception:
        pass
    pred = 'face' if text.upper().startswith('FACE') else ('noface' if 'FACE' in text.upper() or 'NO' in text.upper() else 'unparsed:' + text[:20])
    res.append(dict(crop=m['crop'], label=m['label'], pred=pred, raw=text[:40], p_face=p_face))

json.dump(dict(model=MODEL, load_s=round(t_load,1), infer_s=round(t_inf,2),
               crops_per_s=round(len(man)/t_inf,2), results=res),
          open(OUT, 'w'), indent=1)
print(f'{MODEL}: load {t_load:.0f}s, infer {t_inf:.1f}s ({len(man)/t_inf:.2f} crops/s)')
