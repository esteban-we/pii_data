"""Dump every runtime attribute and tensor shape of the EgoBlur Gen2 jit (CPU, read-only)."""
import sys, json, torch, hashlib
from detectron2.structures import Boxes
from detectron2.export.torchscript_patch import patch_instances
FIELDS = {"proposal_boxes": Boxes, "objectness_logits": torch.Tensor, "pred_boxes": Boxes,
          "scores": torch.Tensor, "pred_classes": torch.Tensor, "pred_masks": torch.Tensor}
P = "/data/esteban/pii/weights/ego_blur_face_gen2.jit"
out = {}
with patch_instances(fields=FIELDS):
    m = torch.jit.load(P, map_location="cpu")
    out["pixel_mean"] = m.pixel_mean.flatten().tolist(); out["pixel_std"] = m.pixel_std.flatten().tolist()
    out["input_format"] = m.input_format
    bb = m.backbone
    out["fpn"] = {"size_divisibility": bb._size_divisibility, "square_pad": bb._square_pad,
                  "top_block": bb.top_block.original_name, "top_in_feature": bb.top_block.in_feature,
                  "in_features": list(bb.in_features) if hasattr(bb, "in_features") else None}
    rpn = m.proposal_generator
    ag = rpn.anchor_generator
    out["rpn"] = {k: getattr(rpn, k) for k in ["batch_size_per_image","positive_fraction","nms_thresh","min_box_size","anchor_boundary_thresh","smooth_l1_beta"]}
    out["rpn"]["pre_nms_topk"] = dict(rpn.pre_nms_topk); out["rpn"]["post_nms_topk"] = dict(rpn.post_nms_topk)
    out["rpn"]["in_features"] = list(rpn.in_features) if hasattr(rpn, "in_features") else None
    out["rpn"]["box2box_weights"] = list(rpn.box2box_transform.weights); out["rpn"]["scale_clamp"] = rpn.box2box_transform.scale_clamp
    out["anchor"] = {"strides": list(ag.strides), "offset": ag.offset,
                     "cell_anchors": [t.tolist() for t in ag.cell_anchors.buffers()]}
    rh = m.roi_heads
    out["roi_heads"] = {k: getattr(rh, k) for k in ["batch_size_per_image","positive_fraction","num_classes","proposal_append_gt","train_on_pred_boxes"]}
    out["roi_heads"]["in_features"] = list(rh.in_features) if hasattr(rh, "in_features") else None
    po = rh.box_pooler
    out["pooler"] = {"output_size": list(po.output_size), "min_level": po.min_level, "max_level": po.max_level,
                     "canonical_level": po.canonical_level, "canonical_box_size": po.canonical_box_size,
                     "level_poolers": [{"scale": l.spatial_scale, "sampling_ratio": l.sampling_ratio, "aligned": l.aligned} for _n, l in po.level_poolers.named_children()]}
    bp = rh.box_predictor
    out["box_predictor"] = {k: getattr(bp, k) for k in ["num_classes","smooth_l1_beta","test_score_thresh","test_nms_thresh","test_topk_per_image","box_reg_loss_type","use_fed_loss","use_sigmoid_ce"]}
    out["box_predictor"]["box2box_weights"] = list(bp.box2box_transform.weights)
    # module classes (norm types, stem, block types)
    classes = {}
    for name, sub in m.named_modules():
        classes[name] = sub.original_name
    out["module_classes"] = classes
    sd = m.state_dict()
    out["state_dict"] = {k: [list(v.shape), str(v.dtype)] for k, v in sd.items()}
    out["n_params_tensors"] = len(sd); out["n_values"] = int(sum(v.numel() for v in sd.values()))
    torch.save({k: v.clone() for k, v in sd.items()}, "/data/esteban/pii/runs/egoblur_build/egoblur_face_gen2_jit_sd.pth")
json.dump(out, open("/data/esteban/pii/runs/egoblur_build/jit_dump.json", "w"), indent=1)
o = out
print("pixel_mean", o["pixel_mean"], "pixel_std", o["pixel_std"], "fmt", o["input_format"])
print("fpn", o["fpn"]); print("rpn", o["rpn"]); print("anchor strides", o["anchor"]["strides"], "offset", o["anchor"]["offset"])
for i, ca in enumerate(o["anchor"]["cell_anchors"]): print("  cell_anchors", i, ca)
print("roi_heads", o["roi_heads"]); print("pooler", o["pooler"]); print("box_predictor", o["box_predictor"])
from collections import Counter
print("norm classes:", Counter(v for k, v in classes.items() if "norm" in k.split(".")[-1] or "Norm" in v))
print("stem:", {k: v for k, v in classes.items() if "stem" in k})
print("res2.0:", {k: v for k, v in classes.items() if k.startswith("backbone.bottom_up.res2.0")})
print("top-level classes:", {k: v for k, v in classes.items() if k.count(".") <= 1})
print("n tensors", o["n_params_tensors"], "n values", o["n_values"])
print("first keys:", list(o["state_dict"])[:8]); print("last keys:", list(o["state_dict"])[-8:])
