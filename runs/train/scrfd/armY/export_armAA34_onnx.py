"""WOR-146: copy of export_armY_onnx.py for armAA34 (CKPT/OUT changed; CFG stays the armY embedded 34G config, whose model block is byte-identical to scrfd_armZ34.py).

Export armY (SCRFD-34G) to ONNX. Copy of pii_train/export_armX_onnx.py (WOR-45)
with CFG switched to the armY 34G config (the ft2 10G template would be the WRONG
architecture), CKPT/OUT updated, and an explicit strict key check added.
CFG is the effective config embedded in epoch_20.pth's meta (verified identical to
training/configs/scrfd_armY.py model+test_cfg blocks). Graph contract is still
diffed against production det_10g.onnx: input name/rank and the nine [N, C]
outputs must match; size/n_nodes will differ (different architecture, expected).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

T = Path("/data/esteban/pii/runs/train/pii_train")
REPO = Path("/data/esteban/pii/runs/train/insightface/detection/scrfd")
CFG = Path("/data/esteban/pii/runs/train/wd_armY_ckpt/embedded_config.py")
CKPT = Path("/data/esteban/pii/runs/train/wd_armAA34/epoch_20.pth")
OUT = Path("/data/esteban/pii/runs/train/armAA34_det34g.onnx")
PROD_ONNX = Path("/data/esteban/pii/weights/det_10g.onnx")

NUM_ANCHORS = 2  # ratios [1.0] x scales [1, 2] in the 34G head, same as 10G

INPUT_NAMES = ["input.1"]
OUTPUT_NAMES = ["score_8", "score_16", "score_32",
                "bbox_8", "bbox_16", "bbox_32",
                "kps_8", "kps_16", "kps_32"]


def install_symbolic_shim() -> None:
    fake = types.ModuleType("mmcv.onnx.symbolic")
    fake.register_extra_symbolics = lambda opset_version=11: None
    sys.modules["mmcv.onnx.symbolic"] = fake
    try:
        import mmcv.onnx as pkg
        pkg.symbolic = fake
    except Exception:
        pkg = types.ModuleType("mmcv.onnx")
        pkg.symbolic = fake
        sys.modules["mmcv.onnx"] = pkg


def describe(path):
    import onnx
    m = onnx.load(str(path))

    def dims(vi):
        return [d.dim_param if d.dim_param else d.dim_value
                for d in vi.type.tensor_type.shape.dim]

    return {"opset": [o.version for o in m.opset_import], "ir": m.ir_version,
            "inputs": [(vi.name, dims(vi)) for vi in m.graph.input],
            "outputs": [(vi.name, dims(vi)) for vi in m.graph.output],
            "n_nodes": len(m.graph.node),
            "size_mb": round(path.stat().st_size / 1e6, 1)}


def main() -> None:
    sys.path.insert(0, str(T))
    sys.path.insert(0, str(REPO))
    import scrfd_shim  # noqa: F401
    install_symbolic_shim()

    import torch
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmdet.models import build_detector

    cfg = Config.fromfile(str(CFG))
    cfg.model.pretrained = None
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg"))

    # Strict key audit: the checkpoint must cover the model exactly.
    ck = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    meta = ck.get("meta", {})
    print(f"ckpt meta: exp_name={meta.get('exp_name')} epoch={meta.get('epoch')} iter={meta.get('iter')}", flush=True)
    ck_keys = set(ck["state_dict"].keys())
    mdl_keys = set(model.state_dict().keys())
    missing = sorted(mdl_keys - ck_keys)
    unexpected = sorted(ck_keys - mdl_keys)
    print(f"key audit: missing={missing} unexpected={unexpected}", flush=True)
    assert not missing and not unexpected, "architecture/checkpoint mismatch"

    load_checkpoint(model, str(CKPT), map_location="cpu")
    model.cpu().eval()

    class ExportWrap(torch.nn.Module):
        def __init__(self, det):
            super().__init__()
            self.det = det

        def forward(self, x):
            outs = self.det.feature_test(x)
            flat = []
            for group in outs:
                flat.extend(group if isinstance(group, (list, tuple)) else [group])
            out = []
            for t in flat:
                if t.dim() == 4:
                    _b, ch, _h, _w = t.shape
                    c = ch // NUM_ANCHORS
                    out.append(t.permute(2, 3, 0, 1).reshape(-1, c))
                else:
                    b, n, c = t.shape
                    out.append(t.reshape(b, n // NUM_ANCHORS, NUM_ANCHORS, c)
                                .permute(1, 0, 2, 3)
                                .reshape(-1, c))
            return tuple(out)

    wrapped = ExportWrap(model).eval()
    dummy = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        probe = wrapped(dummy)
    print(f"wrapped outputs {len(probe)}: {[tuple(t.shape) for t in probe]}", flush=True)
    names = OUTPUT_NAMES[:len(probe)]

    dyn = {n: {0: "?"} for n in names}
    dyn[INPUT_NAMES[0]] = {2: "?", 3: "?"}

    torch.onnx.export(
        wrapped, (dummy,), str(OUT),
        dynamo=False, keep_initializers_as_inputs=False, verbose=False,
        input_names=INPUT_NAMES, output_names=names,
        dynamic_axes=dyn, opset_version=11)
    print(f"exported {OUT.stat().st_size/1e6:.1f} MB -> {OUT}", flush=True)

    print("\n=== graph contract: armY vs production det_10g.onnx ===")
    a, b = describe(OUT), describe(PROD_ONNX)
    for k in ("size_mb", "opset", "ir", "n_nodes"):
        same = a[k] == b[k]
        print(f"  {k:9} armY={a[k]}   prod={b[k]}{'' if same else '   <-- differs'}")
    print(f"  inputs    armY={a['inputs']}")
    print(f"            prod={b['inputs']}"
          f"{'' if a['inputs'] == b['inputs'] else '   <-- differs'}")
    print(f"  n outputs armY={len(a['outputs'])}  prod={len(b['outputs'])}")
    for (na, da), (nb, db) in zip(a["outputs"], b["outputs"]):
        print(f"    {na:10} {str(da):20} | {nb:10} {str(db):20}")
    # Contract check: prod bakes static output dims; ours (like armX's export)
    # carries a dynamic row dim. Require: 9 outputs, same names/order as the
    # head contract, dynamic axis 0, and an input identical to prod.
    assert [n for n, _ in a["outputs"]] == OUTPUT_NAMES, "output names mismatch"
    assert all(d[0] == "?" and len(d) == 2 for _, d in a["outputs"]), "output rank/dyn mismatch"
    assert a["inputs"] == b["inputs"], "input contract mismatch"
    assert a["opset"] == b["opset"] == [11], "opset mismatch"


if __name__ == "__main__":
    main()
