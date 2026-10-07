optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys=dict({
            'backbone.patch_embed.':
            dict(lr_mult=0.013302794647291146),
            'backbone.pos_embed':
            dict(lr_mult=0.013302794647291146, decay_mult=0.0),
            'backbone.cls_token':
            dict(lr_mult=0.013302794647291146, decay_mult=0.0),
            'backbone.register_tokens':
            dict(lr_mult=0.013302794647291146, decay_mult=0.0),
            'backbone.patch_embed.proj.bias':
            dict(lr_mult=0.013302794647291146, decay_mult=0.0),
            'backbone.blocks.0.':
            dict(lr_mult=0.014780882941434608),
            'backbone.blocks.0.norm1.weight':
            dict(lr_mult=0.014780882941434608, decay_mult=0.0),
            'backbone.blocks.0.norm1.bias':
            dict(lr_mult=0.014780882941434608, decay_mult=0.0),
            'backbone.blocks.0.norm2.weight':
            dict(lr_mult=0.014780882941434608, decay_mult=0.0),
            'backbone.blocks.0.norm2.bias':
            dict(lr_mult=0.014780882941434608, decay_mult=0.0),
            'backbone.blocks.0.ls1.gamma':
            dict(lr_mult=0.014780882941434608, decay_mult=0.0),
            'backbone.blocks.0.ls2.gamma':
            dict(lr_mult=0.014780882941434608, decay_mult=0.0),
            'backbone.blocks.0.attn.qkv.bias':
            dict(lr_mult=0.014780882941434608, decay_mult=0.0),
            'backbone.blocks.0.attn.proj.bias':
            dict(lr_mult=0.014780882941434608, decay_mult=0.0),
            'backbone.blocks.0.mlp.w12.bias':
            dict(lr_mult=0.014780882941434608, decay_mult=0.0),
            'backbone.blocks.0.mlp.w3.bias':
            dict(lr_mult=0.014780882941434608, decay_mult=0.0),
            'backbone.blocks.1.':
            dict(lr_mult=0.016423203268260675),
            'backbone.blocks.1.norm1.weight':
            dict(lr_mult=0.016423203268260675, decay_mult=0.0),
            'backbone.blocks.1.norm1.bias':
            dict(lr_mult=0.016423203268260675, decay_mult=0.0),
            'backbone.blocks.1.norm2.weight':
            dict(lr_mult=0.016423203268260675, decay_mult=0.0),
            'backbone.blocks.1.norm2.bias':
            dict(lr_mult=0.016423203268260675, decay_mult=0.0),
            'backbone.blocks.1.ls1.gamma':
            dict(lr_mult=0.016423203268260675, decay_mult=0.0),
            'backbone.blocks.1.ls2.gamma':
            dict(lr_mult=0.016423203268260675, decay_mult=0.0),
            'backbone.blocks.1.attn.qkv.bias':
            dict(lr_mult=0.016423203268260675, decay_mult=0.0),
            'backbone.blocks.1.attn.proj.bias':
            dict(lr_mult=0.016423203268260675, decay_mult=0.0),
            'backbone.blocks.1.mlp.w12.bias':
            dict(lr_mult=0.016423203268260675, decay_mult=0.0),
            'backbone.blocks.1.mlp.w3.bias':
            dict(lr_mult=0.016423203268260675, decay_mult=0.0),
            'backbone.blocks.2.':
            dict(lr_mult=0.01824800363140075),
            'backbone.blocks.2.norm1.weight':
            dict(lr_mult=0.01824800363140075, decay_mult=0.0),
            'backbone.blocks.2.norm1.bias':
            dict(lr_mult=0.01824800363140075, decay_mult=0.0),
            'backbone.blocks.2.norm2.weight':
            dict(lr_mult=0.01824800363140075, decay_mult=0.0),
            'backbone.blocks.2.norm2.bias':
            dict(lr_mult=0.01824800363140075, decay_mult=0.0),
            'backbone.blocks.2.ls1.gamma':
            dict(lr_mult=0.01824800363140075, decay_mult=0.0),
            'backbone.blocks.2.ls2.gamma':
            dict(lr_mult=0.01824800363140075, decay_mult=0.0),
            'backbone.blocks.2.attn.qkv.bias':
            dict(lr_mult=0.01824800363140075, decay_mult=0.0),
            'backbone.blocks.2.attn.proj.bias':
            dict(lr_mult=0.01824800363140075, decay_mult=0.0),
            'backbone.blocks.2.mlp.w12.bias':
            dict(lr_mult=0.01824800363140075, decay_mult=0.0),
            'backbone.blocks.2.mlp.w3.bias':
            dict(lr_mult=0.01824800363140075, decay_mult=0.0),
            'backbone.blocks.3.':
            dict(lr_mult=0.020275559590445275),
            'backbone.blocks.3.norm1.weight':
            dict(lr_mult=0.020275559590445275, decay_mult=0.0),
            'backbone.blocks.3.norm1.bias':
            dict(lr_mult=0.020275559590445275, decay_mult=0.0),
            'backbone.blocks.3.norm2.weight':
            dict(lr_mult=0.020275559590445275, decay_mult=0.0),
            'backbone.blocks.3.norm2.bias':
            dict(lr_mult=0.020275559590445275, decay_mult=0.0),
            'backbone.blocks.3.ls1.gamma':
            dict(lr_mult=0.020275559590445275, decay_mult=0.0),
            'backbone.blocks.3.ls2.gamma':
            dict(lr_mult=0.020275559590445275, decay_mult=0.0),
            'backbone.blocks.3.attn.qkv.bias':
            dict(lr_mult=0.020275559590445275, decay_mult=0.0),
            'backbone.blocks.3.attn.proj.bias':
            dict(lr_mult=0.020275559590445275, decay_mult=0.0),
            'backbone.blocks.3.mlp.w12.bias':
            dict(lr_mult=0.020275559590445275, decay_mult=0.0),
            'backbone.blocks.3.mlp.w3.bias':
            dict(lr_mult=0.020275559590445275, decay_mult=0.0),
            'backbone.blocks.4.':
            dict(lr_mult=0.022528399544939195),
            'backbone.blocks.4.norm1.weight':
            dict(lr_mult=0.022528399544939195, decay_mult=0.0),
            'backbone.blocks.4.norm1.bias':
            dict(lr_mult=0.022528399544939195, decay_mult=0.0),
            'backbone.blocks.4.norm2.weight':
            dict(lr_mult=0.022528399544939195, decay_mult=0.0),
            'backbone.blocks.4.norm2.bias':
            dict(lr_mult=0.022528399544939195, decay_mult=0.0),
            'backbone.blocks.4.ls1.gamma':
            dict(lr_mult=0.022528399544939195, decay_mult=0.0),
            'backbone.blocks.4.ls2.gamma':
            dict(lr_mult=0.022528399544939195, decay_mult=0.0),
            'backbone.blocks.4.attn.qkv.bias':
            dict(lr_mult=0.022528399544939195, decay_mult=0.0),
            'backbone.blocks.4.attn.proj.bias':
            dict(lr_mult=0.022528399544939195, decay_mult=0.0),
            'backbone.blocks.4.mlp.w12.bias':
            dict(lr_mult=0.022528399544939195, decay_mult=0.0),
            'backbone.blocks.4.mlp.w3.bias':
            dict(lr_mult=0.022528399544939195, decay_mult=0.0),
            'backbone.blocks.5.':
            dict(lr_mult=0.025031555049932437),
            'backbone.blocks.5.norm1.weight':
            dict(lr_mult=0.025031555049932437, decay_mult=0.0),
            'backbone.blocks.5.norm1.bias':
            dict(lr_mult=0.025031555049932437, decay_mult=0.0),
            'backbone.blocks.5.norm2.weight':
            dict(lr_mult=0.025031555049932437, decay_mult=0.0),
            'backbone.blocks.5.norm2.bias':
            dict(lr_mult=0.025031555049932437, decay_mult=0.0),
            'backbone.blocks.5.ls1.gamma':
            dict(lr_mult=0.025031555049932437, decay_mult=0.0),
            'backbone.blocks.5.ls2.gamma':
            dict(lr_mult=0.025031555049932437, decay_mult=0.0),
            'backbone.blocks.5.attn.qkv.bias':
            dict(lr_mult=0.025031555049932437, decay_mult=0.0),
            'backbone.blocks.5.attn.proj.bias':
            dict(lr_mult=0.025031555049932437, decay_mult=0.0),
            'backbone.blocks.5.mlp.w12.bias':
            dict(lr_mult=0.025031555049932437, decay_mult=0.0),
            'backbone.blocks.5.mlp.w3.bias':
            dict(lr_mult=0.025031555049932437, decay_mult=0.0),
            'backbone.blocks.6.':
            dict(lr_mult=0.027812838944369374),
            'backbone.blocks.6.norm1.weight':
            dict(lr_mult=0.027812838944369374, decay_mult=0.0),
            'backbone.blocks.6.norm1.bias':
            dict(lr_mult=0.027812838944369374, decay_mult=0.0),
            'backbone.blocks.6.norm2.weight':
            dict(lr_mult=0.027812838944369374, decay_mult=0.0),
            'backbone.blocks.6.norm2.bias':
            dict(lr_mult=0.027812838944369374, decay_mult=0.0),
            'backbone.blocks.6.ls1.gamma':
            dict(lr_mult=0.027812838944369374, decay_mult=0.0),
            'backbone.blocks.6.ls2.gamma':
            dict(lr_mult=0.027812838944369374, decay_mult=0.0),
            'backbone.blocks.6.attn.qkv.bias':
            dict(lr_mult=0.027812838944369374, decay_mult=0.0),
            'backbone.blocks.6.attn.proj.bias':
            dict(lr_mult=0.027812838944369374, decay_mult=0.0),
            'backbone.blocks.6.mlp.w12.bias':
            dict(lr_mult=0.027812838944369374, decay_mult=0.0),
            'backbone.blocks.6.mlp.w3.bias':
            dict(lr_mult=0.027812838944369374, decay_mult=0.0),
            'backbone.blocks.7.':
            dict(lr_mult=0.030903154382632636),
            'backbone.blocks.7.norm1.weight':
            dict(lr_mult=0.030903154382632636, decay_mult=0.0),
            'backbone.blocks.7.norm1.bias':
            dict(lr_mult=0.030903154382632636, decay_mult=0.0),
            'backbone.blocks.7.norm2.weight':
            dict(lr_mult=0.030903154382632636, decay_mult=0.0),
            'backbone.blocks.7.norm2.bias':
            dict(lr_mult=0.030903154382632636, decay_mult=0.0),
            'backbone.blocks.7.ls1.gamma':
            dict(lr_mult=0.030903154382632636, decay_mult=0.0),
            'backbone.blocks.7.ls2.gamma':
            dict(lr_mult=0.030903154382632636, decay_mult=0.0),
            'backbone.blocks.7.attn.qkv.bias':
            dict(lr_mult=0.030903154382632636, decay_mult=0.0),
            'backbone.blocks.7.attn.proj.bias':
            dict(lr_mult=0.030903154382632636, decay_mult=0.0),
            'backbone.blocks.7.mlp.w12.bias':
            dict(lr_mult=0.030903154382632636, decay_mult=0.0),
            'backbone.blocks.7.mlp.w3.bias':
            dict(lr_mult=0.030903154382632636, decay_mult=0.0),
            'backbone.blocks.8.':
            dict(lr_mult=0.03433683820292515),
            'backbone.blocks.8.norm1.weight':
            dict(lr_mult=0.03433683820292515, decay_mult=0.0),
            'backbone.blocks.8.norm1.bias':
            dict(lr_mult=0.03433683820292515, decay_mult=0.0),
            'backbone.blocks.8.norm2.weight':
            dict(lr_mult=0.03433683820292515, decay_mult=0.0),
            'backbone.blocks.8.norm2.bias':
            dict(lr_mult=0.03433683820292515, decay_mult=0.0),
            'backbone.blocks.8.ls1.gamma':
            dict(lr_mult=0.03433683820292515, decay_mult=0.0),
            'backbone.blocks.8.ls2.gamma':
            dict(lr_mult=0.03433683820292515, decay_mult=0.0),
            'backbone.blocks.8.attn.qkv.bias':
            dict(lr_mult=0.03433683820292515, decay_mult=0.0),
            'backbone.blocks.8.attn.proj.bias':
            dict(lr_mult=0.03433683820292515, decay_mult=0.0),
            'backbone.blocks.8.mlp.w12.bias':
            dict(lr_mult=0.03433683820292515, decay_mult=0.0),
            'backbone.blocks.8.mlp.w3.bias':
            dict(lr_mult=0.03433683820292515, decay_mult=0.0),
            'backbone.blocks.9.':
            dict(lr_mult=0.038152042447694615),
            'backbone.blocks.9.norm1.weight':
            dict(lr_mult=0.038152042447694615, decay_mult=0.0),
            'backbone.blocks.9.norm1.bias':
            dict(lr_mult=0.038152042447694615, decay_mult=0.0),
            'backbone.blocks.9.norm2.weight':
            dict(lr_mult=0.038152042447694615, decay_mult=0.0),
            'backbone.blocks.9.norm2.bias':
            dict(lr_mult=0.038152042447694615, decay_mult=0.0),
            'backbone.blocks.9.ls1.gamma':
            dict(lr_mult=0.038152042447694615, decay_mult=0.0),
            'backbone.blocks.9.ls2.gamma':
            dict(lr_mult=0.038152042447694615, decay_mult=0.0),
            'backbone.blocks.9.attn.qkv.bias':
            dict(lr_mult=0.038152042447694615, decay_mult=0.0),
            'backbone.blocks.9.attn.proj.bias':
            dict(lr_mult=0.038152042447694615, decay_mult=0.0),
            'backbone.blocks.9.mlp.w12.bias':
            dict(lr_mult=0.038152042447694615, decay_mult=0.0),
            'backbone.blocks.9.mlp.w3.bias':
            dict(lr_mult=0.038152042447694615, decay_mult=0.0),
            'backbone.blocks.10.':
            dict(lr_mult=0.04239115827521624),
            'backbone.blocks.10.norm1.weight':
            dict(lr_mult=0.04239115827521624, decay_mult=0.0),
            'backbone.blocks.10.norm1.bias':
            dict(lr_mult=0.04239115827521624, decay_mult=0.0),
            'backbone.blocks.10.norm2.weight':
            dict(lr_mult=0.04239115827521624, decay_mult=0.0),
            'backbone.blocks.10.norm2.bias':
            dict(lr_mult=0.04239115827521624, decay_mult=0.0),
            'backbone.blocks.10.ls1.gamma':
            dict(lr_mult=0.04239115827521624, decay_mult=0.0),
            'backbone.blocks.10.ls2.gamma':
            dict(lr_mult=0.04239115827521624, decay_mult=0.0),
            'backbone.blocks.10.attn.qkv.bias':
            dict(lr_mult=0.04239115827521624, decay_mult=0.0),
            'backbone.blocks.10.attn.proj.bias':
            dict(lr_mult=0.04239115827521624, decay_mult=0.0),
            'backbone.blocks.10.mlp.w12.bias':
            dict(lr_mult=0.04239115827521624, decay_mult=0.0),
            'backbone.blocks.10.mlp.w3.bias':
            dict(lr_mult=0.04239115827521624, decay_mult=0.0),
            'backbone.blocks.11.':
            dict(lr_mult=0.047101286972462485),
            'backbone.blocks.11.norm1.weight':
            dict(lr_mult=0.047101286972462485, decay_mult=0.0),
            'backbone.blocks.11.norm1.bias':
            dict(lr_mult=0.047101286972462485, decay_mult=0.0),
            'backbone.blocks.11.norm2.weight':
            dict(lr_mult=0.047101286972462485, decay_mult=0.0),
            'backbone.blocks.11.norm2.bias':
            dict(lr_mult=0.047101286972462485, decay_mult=0.0),
            'backbone.blocks.11.ls1.gamma':
            dict(lr_mult=0.047101286972462485, decay_mult=0.0),
            'backbone.blocks.11.ls2.gamma':
            dict(lr_mult=0.047101286972462485, decay_mult=0.0),
            'backbone.blocks.11.attn.qkv.bias':
            dict(lr_mult=0.047101286972462485, decay_mult=0.0),
            'backbone.blocks.11.attn.proj.bias':
            dict(lr_mult=0.047101286972462485, decay_mult=0.0),
            'backbone.blocks.11.mlp.w12.bias':
            dict(lr_mult=0.047101286972462485, decay_mult=0.0),
            'backbone.blocks.11.mlp.w3.bias':
            dict(lr_mult=0.047101286972462485, decay_mult=0.0),
            'backbone.blocks.12.':
            dict(lr_mult=0.05233476330273609),
            'backbone.blocks.12.norm1.weight':
            dict(lr_mult=0.05233476330273609, decay_mult=0.0),
            'backbone.blocks.12.norm1.bias':
            dict(lr_mult=0.05233476330273609, decay_mult=0.0),
            'backbone.blocks.12.norm2.weight':
            dict(lr_mult=0.05233476330273609, decay_mult=0.0),
            'backbone.blocks.12.norm2.bias':
            dict(lr_mult=0.05233476330273609, decay_mult=0.0),
            'backbone.blocks.12.ls1.gamma':
            dict(lr_mult=0.05233476330273609, decay_mult=0.0),
            'backbone.blocks.12.ls2.gamma':
            dict(lr_mult=0.05233476330273609, decay_mult=0.0),
            'backbone.blocks.12.attn.qkv.bias':
            dict(lr_mult=0.05233476330273609, decay_mult=0.0),
            'backbone.blocks.12.attn.proj.bias':
            dict(lr_mult=0.05233476330273609, decay_mult=0.0),
            'backbone.blocks.12.mlp.w12.bias':
            dict(lr_mult=0.05233476330273609, decay_mult=0.0),
            'backbone.blocks.12.mlp.w3.bias':
            dict(lr_mult=0.05233476330273609, decay_mult=0.0),
            'backbone.blocks.13.':
            dict(lr_mult=0.058149737003040096),
            'backbone.blocks.13.norm1.weight':
            dict(lr_mult=0.058149737003040096, decay_mult=0.0),
            'backbone.blocks.13.norm1.bias':
            dict(lr_mult=0.058149737003040096, decay_mult=0.0),
            'backbone.blocks.13.norm2.weight':
            dict(lr_mult=0.058149737003040096, decay_mult=0.0),
            'backbone.blocks.13.norm2.bias':
            dict(lr_mult=0.058149737003040096, decay_mult=0.0),
            'backbone.blocks.13.ls1.gamma':
            dict(lr_mult=0.058149737003040096, decay_mult=0.0),
            'backbone.blocks.13.ls2.gamma':
            dict(lr_mult=0.058149737003040096, decay_mult=0.0),
            'backbone.blocks.13.attn.qkv.bias':
            dict(lr_mult=0.058149737003040096, decay_mult=0.0),
            'backbone.blocks.13.attn.proj.bias':
            dict(lr_mult=0.058149737003040096, decay_mult=0.0),
            'backbone.blocks.13.mlp.w12.bias':
            dict(lr_mult=0.058149737003040096, decay_mult=0.0),
            'backbone.blocks.13.mlp.w3.bias':
            dict(lr_mult=0.058149737003040096, decay_mult=0.0),
            'backbone.blocks.14.':
            dict(lr_mult=0.06461081889226677),
            'backbone.blocks.14.norm1.weight':
            dict(lr_mult=0.06461081889226677, decay_mult=0.0),
            'backbone.blocks.14.norm1.bias':
            dict(lr_mult=0.06461081889226677, decay_mult=0.0),
            'backbone.blocks.14.norm2.weight':
            dict(lr_mult=0.06461081889226677, decay_mult=0.0),
            'backbone.blocks.14.norm2.bias':
            dict(lr_mult=0.06461081889226677, decay_mult=0.0),
            'backbone.blocks.14.ls1.gamma':
            dict(lr_mult=0.06461081889226677, decay_mult=0.0),
            'backbone.blocks.14.ls2.gamma':
            dict(lr_mult=0.06461081889226677, decay_mult=0.0),
            'backbone.blocks.14.attn.qkv.bias':
            dict(lr_mult=0.06461081889226677, decay_mult=0.0),
            'backbone.blocks.14.attn.proj.bias':
            dict(lr_mult=0.06461081889226677, decay_mult=0.0),
            'backbone.blocks.14.mlp.w12.bias':
            dict(lr_mult=0.06461081889226677, decay_mult=0.0),
            'backbone.blocks.14.mlp.w3.bias':
            dict(lr_mult=0.06461081889226677, decay_mult=0.0),
            'backbone.blocks.15.':
            dict(lr_mult=0.0717897987691853),
            'backbone.blocks.15.norm1.weight':
            dict(lr_mult=0.0717897987691853, decay_mult=0.0),
            'backbone.blocks.15.norm1.bias':
            dict(lr_mult=0.0717897987691853, decay_mult=0.0),
            'backbone.blocks.15.norm2.weight':
            dict(lr_mult=0.0717897987691853, decay_mult=0.0),
            'backbone.blocks.15.norm2.bias':
            dict(lr_mult=0.0717897987691853, decay_mult=0.0),
            'backbone.blocks.15.ls1.gamma':
            dict(lr_mult=0.0717897987691853, decay_mult=0.0),
            'backbone.blocks.15.ls2.gamma':
            dict(lr_mult=0.0717897987691853, decay_mult=0.0),
            'backbone.blocks.15.attn.qkv.bias':
            dict(lr_mult=0.0717897987691853, decay_mult=0.0),
            'backbone.blocks.15.attn.proj.bias':
            dict(lr_mult=0.0717897987691853, decay_mult=0.0),
            'backbone.blocks.15.mlp.w12.bias':
            dict(lr_mult=0.0717897987691853, decay_mult=0.0),
            'backbone.blocks.15.mlp.w3.bias':
            dict(lr_mult=0.0717897987691853, decay_mult=0.0),
            'backbone.blocks.16.':
            dict(lr_mult=0.07976644307687256),
            'backbone.blocks.16.norm1.weight':
            dict(lr_mult=0.07976644307687256, decay_mult=0.0),
            'backbone.blocks.16.norm1.bias':
            dict(lr_mult=0.07976644307687256, decay_mult=0.0),
            'backbone.blocks.16.norm2.weight':
            dict(lr_mult=0.07976644307687256, decay_mult=0.0),
            'backbone.blocks.16.norm2.bias':
            dict(lr_mult=0.07976644307687256, decay_mult=0.0),
            'backbone.blocks.16.ls1.gamma':
            dict(lr_mult=0.07976644307687256, decay_mult=0.0),
            'backbone.blocks.16.ls2.gamma':
            dict(lr_mult=0.07976644307687256, decay_mult=0.0),
            'backbone.blocks.16.attn.qkv.bias':
            dict(lr_mult=0.07976644307687256, decay_mult=0.0),
            'backbone.blocks.16.attn.proj.bias':
            dict(lr_mult=0.07976644307687256, decay_mult=0.0),
            'backbone.blocks.16.mlp.w12.bias':
            dict(lr_mult=0.07976644307687256, decay_mult=0.0),
            'backbone.blocks.16.mlp.w3.bias':
            dict(lr_mult=0.07976644307687256, decay_mult=0.0),
            'backbone.blocks.17.':
            dict(lr_mult=0.08862938119652507),
            'backbone.blocks.17.norm1.weight':
            dict(lr_mult=0.08862938119652507, decay_mult=0.0),
            'backbone.blocks.17.norm1.bias':
            dict(lr_mult=0.08862938119652507, decay_mult=0.0),
            'backbone.blocks.17.norm2.weight':
            dict(lr_mult=0.08862938119652507, decay_mult=0.0),
            'backbone.blocks.17.norm2.bias':
            dict(lr_mult=0.08862938119652507, decay_mult=0.0),
            'backbone.blocks.17.ls1.gamma':
            dict(lr_mult=0.08862938119652507, decay_mult=0.0),
            'backbone.blocks.17.ls2.gamma':
            dict(lr_mult=0.08862938119652507, decay_mult=0.0),
            'backbone.blocks.17.attn.qkv.bias':
            dict(lr_mult=0.08862938119652507, decay_mult=0.0),
            'backbone.blocks.17.attn.proj.bias':
            dict(lr_mult=0.08862938119652507, decay_mult=0.0),
            'backbone.blocks.17.mlp.w12.bias':
            dict(lr_mult=0.08862938119652507, decay_mult=0.0),
            'backbone.blocks.17.mlp.w3.bias':
            dict(lr_mult=0.08862938119652507, decay_mult=0.0),
            'backbone.blocks.18.':
            dict(lr_mult=0.09847709021836118),
            'backbone.blocks.18.norm1.weight':
            dict(lr_mult=0.09847709021836118, decay_mult=0.0),
            'backbone.blocks.18.norm1.bias':
            dict(lr_mult=0.09847709021836118, decay_mult=0.0),
            'backbone.blocks.18.norm2.weight':
            dict(lr_mult=0.09847709021836118, decay_mult=0.0),
            'backbone.blocks.18.norm2.bias':
            dict(lr_mult=0.09847709021836118, decay_mult=0.0),
            'backbone.blocks.18.ls1.gamma':
            dict(lr_mult=0.09847709021836118, decay_mult=0.0),
            'backbone.blocks.18.ls2.gamma':
            dict(lr_mult=0.09847709021836118, decay_mult=0.0),
            'backbone.blocks.18.attn.qkv.bias':
            dict(lr_mult=0.09847709021836118, decay_mult=0.0),
            'backbone.blocks.18.attn.proj.bias':
            dict(lr_mult=0.09847709021836118, decay_mult=0.0),
            'backbone.blocks.18.mlp.w12.bias':
            dict(lr_mult=0.09847709021836118, decay_mult=0.0),
            'backbone.blocks.18.mlp.w3.bias':
            dict(lr_mult=0.09847709021836118, decay_mult=0.0),
            'backbone.blocks.19.':
            dict(lr_mult=0.10941898913151242),
            'backbone.blocks.19.norm1.weight':
            dict(lr_mult=0.10941898913151242, decay_mult=0.0),
            'backbone.blocks.19.norm1.bias':
            dict(lr_mult=0.10941898913151242, decay_mult=0.0),
            'backbone.blocks.19.norm2.weight':
            dict(lr_mult=0.10941898913151242, decay_mult=0.0),
            'backbone.blocks.19.norm2.bias':
            dict(lr_mult=0.10941898913151242, decay_mult=0.0),
            'backbone.blocks.19.ls1.gamma':
            dict(lr_mult=0.10941898913151242, decay_mult=0.0),
            'backbone.blocks.19.ls2.gamma':
            dict(lr_mult=0.10941898913151242, decay_mult=0.0),
            'backbone.blocks.19.attn.qkv.bias':
            dict(lr_mult=0.10941898913151242, decay_mult=0.0),
            'backbone.blocks.19.attn.proj.bias':
            dict(lr_mult=0.10941898913151242, decay_mult=0.0),
            'backbone.blocks.19.mlp.w12.bias':
            dict(lr_mult=0.10941898913151242, decay_mult=0.0),
            'backbone.blocks.19.mlp.w3.bias':
            dict(lr_mult=0.10941898913151242, decay_mult=0.0),
            'backbone.blocks.20.':
            dict(lr_mult=0.12157665459056935),
            'backbone.blocks.20.norm1.weight':
            dict(lr_mult=0.12157665459056935, decay_mult=0.0),
            'backbone.blocks.20.norm1.bias':
            dict(lr_mult=0.12157665459056935, decay_mult=0.0),
            'backbone.blocks.20.norm2.weight':
            dict(lr_mult=0.12157665459056935, decay_mult=0.0),
            'backbone.blocks.20.norm2.bias':
            dict(lr_mult=0.12157665459056935, decay_mult=0.0),
            'backbone.blocks.20.ls1.gamma':
            dict(lr_mult=0.12157665459056935, decay_mult=0.0),
            'backbone.blocks.20.ls2.gamma':
            dict(lr_mult=0.12157665459056935, decay_mult=0.0),
            'backbone.blocks.20.attn.qkv.bias':
            dict(lr_mult=0.12157665459056935, decay_mult=0.0),
            'backbone.blocks.20.attn.proj.bias':
            dict(lr_mult=0.12157665459056935, decay_mult=0.0),
            'backbone.blocks.20.mlp.w12.bias':
            dict(lr_mult=0.12157665459056935, decay_mult=0.0),
            'backbone.blocks.20.mlp.w3.bias':
            dict(lr_mult=0.12157665459056935, decay_mult=0.0),
            'backbone.blocks.21.':
            dict(lr_mult=0.13508517176729928),
            'backbone.blocks.21.norm1.weight':
            dict(lr_mult=0.13508517176729928, decay_mult=0.0),
            'backbone.blocks.21.norm1.bias':
            dict(lr_mult=0.13508517176729928, decay_mult=0.0),
            'backbone.blocks.21.norm2.weight':
            dict(lr_mult=0.13508517176729928, decay_mult=0.0),
            'backbone.blocks.21.norm2.bias':
            dict(lr_mult=0.13508517176729928, decay_mult=0.0),
            'backbone.blocks.21.ls1.gamma':
            dict(lr_mult=0.13508517176729928, decay_mult=0.0),
            'backbone.blocks.21.ls2.gamma':
            dict(lr_mult=0.13508517176729928, decay_mult=0.0),
            'backbone.blocks.21.attn.qkv.bias':
            dict(lr_mult=0.13508517176729928, decay_mult=0.0),
            'backbone.blocks.21.attn.proj.bias':
            dict(lr_mult=0.13508517176729928, decay_mult=0.0),
            'backbone.blocks.21.mlp.w12.bias':
            dict(lr_mult=0.13508517176729928, decay_mult=0.0),
            'backbone.blocks.21.mlp.w3.bias':
            dict(lr_mult=0.13508517176729928, decay_mult=0.0),
            'backbone.blocks.22.':
            dict(lr_mult=0.15009463529699918),
            'backbone.blocks.22.norm1.weight':
            dict(lr_mult=0.15009463529699918, decay_mult=0.0),
            'backbone.blocks.22.norm1.bias':
            dict(lr_mult=0.15009463529699918, decay_mult=0.0),
            'backbone.blocks.22.norm2.weight':
            dict(lr_mult=0.15009463529699918, decay_mult=0.0),
            'backbone.blocks.22.norm2.bias':
            dict(lr_mult=0.15009463529699918, decay_mult=0.0),
            'backbone.blocks.22.ls1.gamma':
            dict(lr_mult=0.15009463529699918, decay_mult=0.0),
            'backbone.blocks.22.ls2.gamma':
            dict(lr_mult=0.15009463529699918, decay_mult=0.0),
            'backbone.blocks.22.attn.qkv.bias':
            dict(lr_mult=0.15009463529699918, decay_mult=0.0),
            'backbone.blocks.22.attn.proj.bias':
            dict(lr_mult=0.15009463529699918, decay_mult=0.0),
            'backbone.blocks.22.mlp.w12.bias':
            dict(lr_mult=0.15009463529699918, decay_mult=0.0),
            'backbone.blocks.22.mlp.w3.bias':
            dict(lr_mult=0.15009463529699918, decay_mult=0.0),
            'backbone.blocks.23.':
            dict(lr_mult=0.16677181699666577),
            'backbone.blocks.23.norm1.weight':
            dict(lr_mult=0.16677181699666577, decay_mult=0.0),
            'backbone.blocks.23.norm1.bias':
            dict(lr_mult=0.16677181699666577, decay_mult=0.0),
            'backbone.blocks.23.norm2.weight':
            dict(lr_mult=0.16677181699666577, decay_mult=0.0),
            'backbone.blocks.23.norm2.bias':
            dict(lr_mult=0.16677181699666577, decay_mult=0.0),
            'backbone.blocks.23.ls1.gamma':
            dict(lr_mult=0.16677181699666577, decay_mult=0.0),
            'backbone.blocks.23.ls2.gamma':
            dict(lr_mult=0.16677181699666577, decay_mult=0.0),
            'backbone.blocks.23.attn.qkv.bias':
            dict(lr_mult=0.16677181699666577, decay_mult=0.0),
            'backbone.blocks.23.attn.proj.bias':
            dict(lr_mult=0.16677181699666577, decay_mult=0.0),
            'backbone.blocks.23.mlp.w12.bias':
            dict(lr_mult=0.16677181699666577, decay_mult=0.0),
            'backbone.blocks.23.mlp.w3.bias':
            dict(lr_mult=0.16677181699666577, decay_mult=0.0),
            'backbone.blocks.24.':
            dict(lr_mult=0.18530201888518416),
            'backbone.blocks.24.norm1.weight':
            dict(lr_mult=0.18530201888518416, decay_mult=0.0),
            'backbone.blocks.24.norm1.bias':
            dict(lr_mult=0.18530201888518416, decay_mult=0.0),
            'backbone.blocks.24.norm2.weight':
            dict(lr_mult=0.18530201888518416, decay_mult=0.0),
            'backbone.blocks.24.norm2.bias':
            dict(lr_mult=0.18530201888518416, decay_mult=0.0),
            'backbone.blocks.24.ls1.gamma':
            dict(lr_mult=0.18530201888518416, decay_mult=0.0),
            'backbone.blocks.24.ls2.gamma':
            dict(lr_mult=0.18530201888518416, decay_mult=0.0),
            'backbone.blocks.24.attn.qkv.bias':
            dict(lr_mult=0.18530201888518416, decay_mult=0.0),
            'backbone.blocks.24.attn.proj.bias':
            dict(lr_mult=0.18530201888518416, decay_mult=0.0),
            'backbone.blocks.24.mlp.w12.bias':
            dict(lr_mult=0.18530201888518416, decay_mult=0.0),
            'backbone.blocks.24.mlp.w3.bias':
            dict(lr_mult=0.18530201888518416, decay_mult=0.0),
            'backbone.blocks.25.':
            dict(lr_mult=0.20589113209464907),
            'backbone.blocks.25.norm1.weight':
            dict(lr_mult=0.20589113209464907, decay_mult=0.0),
            'backbone.blocks.25.norm1.bias':
            dict(lr_mult=0.20589113209464907, decay_mult=0.0),
            'backbone.blocks.25.norm2.weight':
            dict(lr_mult=0.20589113209464907, decay_mult=0.0),
            'backbone.blocks.25.norm2.bias':
            dict(lr_mult=0.20589113209464907, decay_mult=0.0),
            'backbone.blocks.25.ls1.gamma':
            dict(lr_mult=0.20589113209464907, decay_mult=0.0),
            'backbone.blocks.25.ls2.gamma':
            dict(lr_mult=0.20589113209464907, decay_mult=0.0),
            'backbone.blocks.25.attn.qkv.bias':
            dict(lr_mult=0.20589113209464907, decay_mult=0.0),
            'backbone.blocks.25.attn.proj.bias':
            dict(lr_mult=0.20589113209464907, decay_mult=0.0),
            'backbone.blocks.25.mlp.w12.bias':
            dict(lr_mult=0.20589113209464907, decay_mult=0.0),
            'backbone.blocks.25.mlp.w3.bias':
            dict(lr_mult=0.20589113209464907, decay_mult=0.0),
            'backbone.blocks.26.':
            dict(lr_mult=0.2287679245496101),
            'backbone.blocks.26.norm1.weight':
            dict(lr_mult=0.2287679245496101, decay_mult=0.0),
            'backbone.blocks.26.norm1.bias':
            dict(lr_mult=0.2287679245496101, decay_mult=0.0),
            'backbone.blocks.26.norm2.weight':
            dict(lr_mult=0.2287679245496101, decay_mult=0.0),
            'backbone.blocks.26.norm2.bias':
            dict(lr_mult=0.2287679245496101, decay_mult=0.0),
            'backbone.blocks.26.ls1.gamma':
            dict(lr_mult=0.2287679245496101, decay_mult=0.0),
            'backbone.blocks.26.ls2.gamma':
            dict(lr_mult=0.2287679245496101, decay_mult=0.0),
            'backbone.blocks.26.attn.qkv.bias':
            dict(lr_mult=0.2287679245496101, decay_mult=0.0),
            'backbone.blocks.26.attn.proj.bias':
            dict(lr_mult=0.2287679245496101, decay_mult=0.0),
            'backbone.blocks.26.mlp.w12.bias':
            dict(lr_mult=0.2287679245496101, decay_mult=0.0),
            'backbone.blocks.26.mlp.w3.bias':
            dict(lr_mult=0.2287679245496101, decay_mult=0.0),
            'backbone.blocks.27.':
            dict(lr_mult=0.2541865828329001),
            'backbone.blocks.27.norm1.weight':
            dict(lr_mult=0.2541865828329001, decay_mult=0.0),
            'backbone.blocks.27.norm1.bias':
            dict(lr_mult=0.2541865828329001, decay_mult=0.0),
            'backbone.blocks.27.norm2.weight':
            dict(lr_mult=0.2541865828329001, decay_mult=0.0),
            'backbone.blocks.27.norm2.bias':
            dict(lr_mult=0.2541865828329001, decay_mult=0.0),
            'backbone.blocks.27.ls1.gamma':
            dict(lr_mult=0.2541865828329001, decay_mult=0.0),
            'backbone.blocks.27.ls2.gamma':
            dict(lr_mult=0.2541865828329001, decay_mult=0.0),
            'backbone.blocks.27.attn.qkv.bias':
            dict(lr_mult=0.2541865828329001, decay_mult=0.0),
            'backbone.blocks.27.attn.proj.bias':
            dict(lr_mult=0.2541865828329001, decay_mult=0.0),
            'backbone.blocks.27.mlp.w12.bias':
            dict(lr_mult=0.2541865828329001, decay_mult=0.0),
            'backbone.blocks.27.mlp.w3.bias':
            dict(lr_mult=0.2541865828329001, decay_mult=0.0),
            'backbone.blocks.28.':
            dict(lr_mult=0.2824295364810001),
            'backbone.blocks.28.norm1.weight':
            dict(lr_mult=0.2824295364810001, decay_mult=0.0),
            'backbone.blocks.28.norm1.bias':
            dict(lr_mult=0.2824295364810001, decay_mult=0.0),
            'backbone.blocks.28.norm2.weight':
            dict(lr_mult=0.2824295364810001, decay_mult=0.0),
            'backbone.blocks.28.norm2.bias':
            dict(lr_mult=0.2824295364810001, decay_mult=0.0),
            'backbone.blocks.28.ls1.gamma':
            dict(lr_mult=0.2824295364810001, decay_mult=0.0),
            'backbone.blocks.28.ls2.gamma':
            dict(lr_mult=0.2824295364810001, decay_mult=0.0),
            'backbone.blocks.28.attn.qkv.bias':
            dict(lr_mult=0.2824295364810001, decay_mult=0.0),
            'backbone.blocks.28.attn.proj.bias':
            dict(lr_mult=0.2824295364810001, decay_mult=0.0),
            'backbone.blocks.28.mlp.w12.bias':
            dict(lr_mult=0.2824295364810001, decay_mult=0.0),
            'backbone.blocks.28.mlp.w3.bias':
            dict(lr_mult=0.2824295364810001, decay_mult=0.0),
            'backbone.blocks.29.':
            dict(lr_mult=0.31381059609000006),
            'backbone.blocks.29.norm1.weight':
            dict(lr_mult=0.31381059609000006, decay_mult=0.0),
            'backbone.blocks.29.norm1.bias':
            dict(lr_mult=0.31381059609000006, decay_mult=0.0),
            'backbone.blocks.29.norm2.weight':
            dict(lr_mult=0.31381059609000006, decay_mult=0.0),
            'backbone.blocks.29.norm2.bias':
            dict(lr_mult=0.31381059609000006, decay_mult=0.0),
            'backbone.blocks.29.ls1.gamma':
            dict(lr_mult=0.31381059609000006, decay_mult=0.0),
            'backbone.blocks.29.ls2.gamma':
            dict(lr_mult=0.31381059609000006, decay_mult=0.0),
            'backbone.blocks.29.attn.qkv.bias':
            dict(lr_mult=0.31381059609000006, decay_mult=0.0),
            'backbone.blocks.29.attn.proj.bias':
            dict(lr_mult=0.31381059609000006, decay_mult=0.0),
            'backbone.blocks.29.mlp.w12.bias':
            dict(lr_mult=0.31381059609000006, decay_mult=0.0),
            'backbone.blocks.29.mlp.w3.bias':
            dict(lr_mult=0.31381059609000006, decay_mult=0.0),
            'backbone.blocks.30.':
            dict(lr_mult=0.3486784401000001),
            'backbone.blocks.30.norm1.weight':
            dict(lr_mult=0.3486784401000001, decay_mult=0.0),
            'backbone.blocks.30.norm1.bias':
            dict(lr_mult=0.3486784401000001, decay_mult=0.0),
            'backbone.blocks.30.norm2.weight':
            dict(lr_mult=0.3486784401000001, decay_mult=0.0),
            'backbone.blocks.30.norm2.bias':
            dict(lr_mult=0.3486784401000001, decay_mult=0.0),
            'backbone.blocks.30.ls1.gamma':
            dict(lr_mult=0.3486784401000001, decay_mult=0.0),
            'backbone.blocks.30.ls2.gamma':
            dict(lr_mult=0.3486784401000001, decay_mult=0.0),
            'backbone.blocks.30.attn.qkv.bias':
            dict(lr_mult=0.3486784401000001, decay_mult=0.0),
            'backbone.blocks.30.attn.proj.bias':
            dict(lr_mult=0.3486784401000001, decay_mult=0.0),
            'backbone.blocks.30.mlp.w12.bias':
            dict(lr_mult=0.3486784401000001, decay_mult=0.0),
            'backbone.blocks.30.mlp.w3.bias':
            dict(lr_mult=0.3486784401000001, decay_mult=0.0),
            'backbone.blocks.31.':
            dict(lr_mult=0.3874204890000001),
            'backbone.blocks.31.norm1.weight':
            dict(lr_mult=0.3874204890000001, decay_mult=0.0),
            'backbone.blocks.31.norm1.bias':
            dict(lr_mult=0.3874204890000001, decay_mult=0.0),
            'backbone.blocks.31.norm2.weight':
            dict(lr_mult=0.3874204890000001, decay_mult=0.0),
            'backbone.blocks.31.norm2.bias':
            dict(lr_mult=0.3874204890000001, decay_mult=0.0),
            'backbone.blocks.31.ls1.gamma':
            dict(lr_mult=0.3874204890000001, decay_mult=0.0),
            'backbone.blocks.31.ls2.gamma':
            dict(lr_mult=0.3874204890000001, decay_mult=0.0),
            'backbone.blocks.31.attn.qkv.bias':
            dict(lr_mult=0.3874204890000001, decay_mult=0.0),
            'backbone.blocks.31.attn.proj.bias':
            dict(lr_mult=0.3874204890000001, decay_mult=0.0),
            'backbone.blocks.31.mlp.w12.bias':
            dict(lr_mult=0.3874204890000001, decay_mult=0.0),
            'backbone.blocks.31.mlp.w3.bias':
            dict(lr_mult=0.3874204890000001, decay_mult=0.0),
            'backbone.blocks.32.':
            dict(lr_mult=0.4304672100000001),
            'backbone.blocks.32.norm1.weight':
            dict(lr_mult=0.4304672100000001, decay_mult=0.0),
            'backbone.blocks.32.norm1.bias':
            dict(lr_mult=0.4304672100000001, decay_mult=0.0),
            'backbone.blocks.32.norm2.weight':
            dict(lr_mult=0.4304672100000001, decay_mult=0.0),
            'backbone.blocks.32.norm2.bias':
            dict(lr_mult=0.4304672100000001, decay_mult=0.0),
            'backbone.blocks.32.ls1.gamma':
            dict(lr_mult=0.4304672100000001, decay_mult=0.0),
            'backbone.blocks.32.ls2.gamma':
            dict(lr_mult=0.4304672100000001, decay_mult=0.0),
            'backbone.blocks.32.attn.qkv.bias':
            dict(lr_mult=0.4304672100000001, decay_mult=0.0),
            'backbone.blocks.32.attn.proj.bias':
            dict(lr_mult=0.4304672100000001, decay_mult=0.0),
            'backbone.blocks.32.mlp.w12.bias':
            dict(lr_mult=0.4304672100000001, decay_mult=0.0),
            'backbone.blocks.32.mlp.w3.bias':
            dict(lr_mult=0.4304672100000001, decay_mult=0.0),
            'backbone.blocks.33.':
            dict(lr_mult=0.4782969000000001),
            'backbone.blocks.33.norm1.weight':
            dict(lr_mult=0.4782969000000001, decay_mult=0.0),
            'backbone.blocks.33.norm1.bias':
            dict(lr_mult=0.4782969000000001, decay_mult=0.0),
            'backbone.blocks.33.norm2.weight':
            dict(lr_mult=0.4782969000000001, decay_mult=0.0),
            'backbone.blocks.33.norm2.bias':
            dict(lr_mult=0.4782969000000001, decay_mult=0.0),
            'backbone.blocks.33.ls1.gamma':
            dict(lr_mult=0.4782969000000001, decay_mult=0.0),
            'backbone.blocks.33.ls2.gamma':
            dict(lr_mult=0.4782969000000001, decay_mult=0.0),
            'backbone.blocks.33.attn.qkv.bias':
            dict(lr_mult=0.4782969000000001, decay_mult=0.0),
            'backbone.blocks.33.attn.proj.bias':
            dict(lr_mult=0.4782969000000001, decay_mult=0.0),
            'backbone.blocks.33.mlp.w12.bias':
            dict(lr_mult=0.4782969000000001, decay_mult=0.0),
            'backbone.blocks.33.mlp.w3.bias':
            dict(lr_mult=0.4782969000000001, decay_mult=0.0),
            'backbone.blocks.34.':
            dict(lr_mult=0.531441),
            'backbone.blocks.34.norm1.weight':
            dict(lr_mult=0.531441, decay_mult=0.0),
            'backbone.blocks.34.norm1.bias':
            dict(lr_mult=0.531441, decay_mult=0.0),
            'backbone.blocks.34.norm2.weight':
            dict(lr_mult=0.531441, decay_mult=0.0),
            'backbone.blocks.34.norm2.bias':
            dict(lr_mult=0.531441, decay_mult=0.0),
            'backbone.blocks.34.ls1.gamma':
            dict(lr_mult=0.531441, decay_mult=0.0),
            'backbone.blocks.34.ls2.gamma':
            dict(lr_mult=0.531441, decay_mult=0.0),
            'backbone.blocks.34.attn.qkv.bias':
            dict(lr_mult=0.531441, decay_mult=0.0),
            'backbone.blocks.34.attn.proj.bias':
            dict(lr_mult=0.531441, decay_mult=0.0),
            'backbone.blocks.34.mlp.w12.bias':
            dict(lr_mult=0.531441, decay_mult=0.0),
            'backbone.blocks.34.mlp.w3.bias':
            dict(lr_mult=0.531441, decay_mult=0.0),
            'backbone.blocks.35.':
            dict(lr_mult=0.5904900000000001),
            'backbone.blocks.35.norm1.weight':
            dict(lr_mult=0.5904900000000001, decay_mult=0.0),
            'backbone.blocks.35.norm1.bias':
            dict(lr_mult=0.5904900000000001, decay_mult=0.0),
            'backbone.blocks.35.norm2.weight':
            dict(lr_mult=0.5904900000000001, decay_mult=0.0),
            'backbone.blocks.35.norm2.bias':
            dict(lr_mult=0.5904900000000001, decay_mult=0.0),
            'backbone.blocks.35.ls1.gamma':
            dict(lr_mult=0.5904900000000001, decay_mult=0.0),
            'backbone.blocks.35.ls2.gamma':
            dict(lr_mult=0.5904900000000001, decay_mult=0.0),
            'backbone.blocks.35.attn.qkv.bias':
            dict(lr_mult=0.5904900000000001, decay_mult=0.0),
            'backbone.blocks.35.attn.proj.bias':
            dict(lr_mult=0.5904900000000001, decay_mult=0.0),
            'backbone.blocks.35.mlp.w12.bias':
            dict(lr_mult=0.5904900000000001, decay_mult=0.0),
            'backbone.blocks.35.mlp.w3.bias':
            dict(lr_mult=0.5904900000000001, decay_mult=0.0),
            'backbone.blocks.36.':
            dict(lr_mult=0.6561),
            'backbone.blocks.36.norm1.weight':
            dict(lr_mult=0.6561, decay_mult=0.0),
            'backbone.blocks.36.norm1.bias':
            dict(lr_mult=0.6561, decay_mult=0.0),
            'backbone.blocks.36.norm2.weight':
            dict(lr_mult=0.6561, decay_mult=0.0),
            'backbone.blocks.36.norm2.bias':
            dict(lr_mult=0.6561, decay_mult=0.0),
            'backbone.blocks.36.ls1.gamma':
            dict(lr_mult=0.6561, decay_mult=0.0),
            'backbone.blocks.36.ls2.gamma':
            dict(lr_mult=0.6561, decay_mult=0.0),
            'backbone.blocks.36.attn.qkv.bias':
            dict(lr_mult=0.6561, decay_mult=0.0),
            'backbone.blocks.36.attn.proj.bias':
            dict(lr_mult=0.6561, decay_mult=0.0),
            'backbone.blocks.36.mlp.w12.bias':
            dict(lr_mult=0.6561, decay_mult=0.0),
            'backbone.blocks.36.mlp.w3.bias':
            dict(lr_mult=0.6561, decay_mult=0.0),
            'backbone.blocks.37.':
            dict(lr_mult=0.7290000000000001),
            'backbone.blocks.37.norm1.weight':
            dict(lr_mult=0.7290000000000001, decay_mult=0.0),
            'backbone.blocks.37.norm1.bias':
            dict(lr_mult=0.7290000000000001, decay_mult=0.0),
            'backbone.blocks.37.norm2.weight':
            dict(lr_mult=0.7290000000000001, decay_mult=0.0),
            'backbone.blocks.37.norm2.bias':
            dict(lr_mult=0.7290000000000001, decay_mult=0.0),
            'backbone.blocks.37.ls1.gamma':
            dict(lr_mult=0.7290000000000001, decay_mult=0.0),
            'backbone.blocks.37.ls2.gamma':
            dict(lr_mult=0.7290000000000001, decay_mult=0.0),
            'backbone.blocks.37.attn.qkv.bias':
            dict(lr_mult=0.7290000000000001, decay_mult=0.0),
            'backbone.blocks.37.attn.proj.bias':
            dict(lr_mult=0.7290000000000001, decay_mult=0.0),
            'backbone.blocks.37.mlp.w12.bias':
            dict(lr_mult=0.7290000000000001, decay_mult=0.0),
            'backbone.blocks.37.mlp.w3.bias':
            dict(lr_mult=0.7290000000000001, decay_mult=0.0),
            'backbone.blocks.38.':
            dict(lr_mult=0.81),
            'backbone.blocks.38.norm1.weight':
            dict(lr_mult=0.81, decay_mult=0.0),
            'backbone.blocks.38.norm1.bias':
            dict(lr_mult=0.81, decay_mult=0.0),
            'backbone.blocks.38.norm2.weight':
            dict(lr_mult=0.81, decay_mult=0.0),
            'backbone.blocks.38.norm2.bias':
            dict(lr_mult=0.81, decay_mult=0.0),
            'backbone.blocks.38.ls1.gamma':
            dict(lr_mult=0.81, decay_mult=0.0),
            'backbone.blocks.38.ls2.gamma':
            dict(lr_mult=0.81, decay_mult=0.0),
            'backbone.blocks.38.attn.qkv.bias':
            dict(lr_mult=0.81, decay_mult=0.0),
            'backbone.blocks.38.attn.proj.bias':
            dict(lr_mult=0.81, decay_mult=0.0),
            'backbone.blocks.38.mlp.w12.bias':
            dict(lr_mult=0.81, decay_mult=0.0),
            'backbone.blocks.38.mlp.w3.bias':
            dict(lr_mult=0.81, decay_mult=0.0),
            'backbone.blocks.39.':
            dict(lr_mult=0.9),
            'backbone.blocks.39.norm1.weight':
            dict(lr_mult=0.9, decay_mult=0.0),
            'backbone.blocks.39.norm1.bias':
            dict(lr_mult=0.9, decay_mult=0.0),
            'backbone.blocks.39.norm2.weight':
            dict(lr_mult=0.9, decay_mult=0.0),
            'backbone.blocks.39.norm2.bias':
            dict(lr_mult=0.9, decay_mult=0.0),
            'backbone.blocks.39.ls1.gamma':
            dict(lr_mult=0.9, decay_mult=0.0),
            'backbone.blocks.39.ls2.gamma':
            dict(lr_mult=0.9, decay_mult=0.0),
            'backbone.blocks.39.attn.qkv.bias':
            dict(lr_mult=0.9, decay_mult=0.0),
            'backbone.blocks.39.attn.proj.bias':
            dict(lr_mult=0.9, decay_mult=0.0),
            'backbone.blocks.39.mlp.w12.bias':
            dict(lr_mult=0.9, decay_mult=0.0),
            'backbone.blocks.39.mlp.w3.bias':
            dict(lr_mult=0.9, decay_mult=0.0),
            'backbone.norm.':
            dict(lr_mult=1.0, decay_mult=0.0)
        }),
        norm_decay_mult=0.0,
        bias_decay_mult=0.0))
optimizer_config = dict(
    type='GradientCumulativeOptimizerHook',
    cumulative_iters=1,
    grad_clip=dict(max_norm=1.0, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[21, 27])
total_epochs = 30
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
work_dir = '/mnt/cachefs/esteban/pii/runs/train/wd_armAU_fluence'
resume_from = None
workflow = [('train', 1)]
dataset_type = 'RetinaFaceDataset'
data_root = 'data/retinaface/'
train_root = 'data/retinaface/train/'
val_root = 'data/retinaface/val/'
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_keypoints=True),
    dict(
        type='RandomSquareCrop',
        crop_choice=[0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        bbox_clip_border=False),
    dict(
        type='Resize',
        img_scale=(672, 672),
        keep_ratio=False,
        bbox_clip_border=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[128.0, 128.0, 128.0],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore',
            'gt_keypointss'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[128.0, 128.0, 128.0],
                to_rgb=True),
            dict(type='Pad', size=(640, 640), pad_val=0),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=16,
    train=dict(
        type='RetinaFaceDataset',
        ann_file='data/mix_ds/train_Z6.txt',
        img_prefix='data/mix_ds/images/',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True, with_keypoints=True),
            dict(
                type='RandomSquareCrop',
                crop_choice=[
                    0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0
                ],
                bbox_clip_border=False),
            dict(
                type='Resize',
                img_scale=(672, 672),
                keep_ratio=False,
                bbox_clip_border=False),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[128.0, 128.0, 128.0],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=[
                    'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore',
                    'gt_keypointss'
                ])
        ]),
    val=dict(
        type='RetinaFaceDataset',
        ann_file='/data/liangzhenghao/face_pii/raw_ds/val/labelv2.txt',
        img_prefix='/data/liangzhenghao/face_pii/raw_ds/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=0.0),
                    dict(
                        type='Normalize',
                        mean=[127.5, 127.5, 127.5],
                        std=[128.0, 128.0, 128.0],
                        to_rgb=True),
                    dict(type='Pad', size=(640, 640), pad_val=0),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='RetinaFaceDataset',
        ann_file='/data/liangzhenghao/face_pii/raw_ds/val/labelv2.txt',
        img_prefix='/data/liangzhenghao/face_pii/raw_ds/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=0.0),
                    dict(
                        type='Normalize',
                        mean=[127.5, 127.5, 127.5],
                        std=[128.0, 128.0, 128.0],
                        to_rgb=True),
                    dict(type='Pad', size=(640, 640), pad_val=0),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
train_cfg = dict(
    assigner=dict(type='ATSSAssigner', topk=9, ignore_iof_thr=0.5),
    allowed_border=-1,
    pos_weight=-1,
    debug=False,
    neg_weight_zero_prefix='/pii',
    neg_weight_full_substrings=['BTJPSP', 'NQYJPB'],
    neg_weight_pii=0.5,
    qscore='one')
test_cfg = dict(
    nms_pre=-1,
    min_bbox_size=0,
    score_thr=0.02,
    nms=dict(type='nms', iou_threshold=0.45),
    max_per_img=-1)
model = dict(
    type='SCRFD',
    backbone=dict(
        type='DINOv2ViT',
        arch='vitg14',
        patch_size=14,
        registers=True,
        drop_path_rate=0.4,
        out_indices=(9, 19, 29, 39),
        pretrained=
        '/mnt/cachefs/esteban/pii/weights/dinov2_vitg14_reg4_pretrain.pth',
        bf16=True,
        use_checkpoint=False),
    neck=dict(
        type='SimpleFPN14',
        in_channels=1536,
        out_channels=128,
        num_inputs=4,
        fuse_inputs=False),
    bbox_head=dict(
        type='SCRFDHead',
        num_classes=1,
        in_channels=128,
        stacked_convs=2,
        feat_channels=256,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        cls_reg_share=True,
        strides_share=True,
        scale_mode=2,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales=[1, 2],
            base_sizes=[16, 64, 256],
            strides=[8, 16, 32]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=False,
        reg_max=8,
        loss_bbox=dict(type='DIoULoss', loss_weight=2.0),
        use_kps=True,
        loss_kps=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=0.1),
        train_cfg=dict(
            assigner=dict(type='ATSSAssigner', topk=9, ignore_iof_thr=0.5),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
            neg_weight_zero_prefix='/pii',
            neg_weight_full_substrings=['BTJPSP', 'NQYJPB'],
            neg_weight_pii=0.5,
            qscore='one'),
        test_cfg=dict(
            nms_pre=-1,
            min_bbox_size=0,
            score_thr=0.02,
            nms=dict(type='nms', iou_threshold=0.45),
            max_per_img=-1)))
evaluation = dict(interval=10000, metric='mAP')
custom_hooks = [
    dict(
        type='EMAHook', momentum=0.0002, interval=1, warm_up=100, priority=49)
]
custom_imports = dict(
    imports=['dinov2_backbone', 'simple_fpn'], allow_failed_imports=False)
gpu_ids = range(0, 8)
