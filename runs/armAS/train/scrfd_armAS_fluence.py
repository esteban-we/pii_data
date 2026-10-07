optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys=dict({
            'backbone.patch_embed.':
            dict(lr_mult=0.0037778931862957215),
            'backbone.cls_token':
            dict(lr_mult=0.0037778931862957215, decay_mult=0.0),
            'backbone.storage_tokens':
            dict(lr_mult=0.0037778931862957215, decay_mult=0.0),
            'backbone.patch_embed.projection.bias':
            dict(lr_mult=0.0037778931862957215, decay_mult=0.0),
            'backbone.blocks.0.':
            dict(lr_mult=0.004722366482869652),
            'backbone.blocks.0.ln1.weight':
            dict(lr_mult=0.004722366482869652, decay_mult=0.0),
            'backbone.blocks.0.ln2.weight':
            dict(lr_mult=0.004722366482869652, decay_mult=0.0),
            'backbone.blocks.0.attn.q_norm.weight':
            dict(lr_mult=0.004722366482869652, decay_mult=0.0),
            'backbone.blocks.0.attn.k_norm.weight':
            dict(lr_mult=0.004722366482869652, decay_mult=0.0),
            'backbone.blocks.0.attn.gamma.weight':
            dict(lr_mult=0.004722366482869652, decay_mult=0.0),
            'backbone.blocks.0.attn.wq.bias':
            dict(lr_mult=0.004722366482869652, decay_mult=0.0),
            'backbone.blocks.0.attn.wk.bias':
            dict(lr_mult=0.004722366482869652, decay_mult=0.0),
            'backbone.blocks.0.attn.wv.bias':
            dict(lr_mult=0.004722366482869652, decay_mult=0.0),
            'backbone.blocks.0.attn.proj.bias':
            dict(lr_mult=0.004722366482869652, decay_mult=0.0),
            'backbone.blocks.0.ffn.w12.bias':
            dict(lr_mult=0.004722366482869652, decay_mult=0.0),
            'backbone.blocks.0.ffn.w3.bias':
            dict(lr_mult=0.004722366482869652, decay_mult=0.0),
            'backbone.blocks.1.':
            dict(lr_mult=0.005902958103587064),
            'backbone.blocks.1.ln1.weight':
            dict(lr_mult=0.005902958103587064, decay_mult=0.0),
            'backbone.blocks.1.ln2.weight':
            dict(lr_mult=0.005902958103587064, decay_mult=0.0),
            'backbone.blocks.1.attn.q_norm.weight':
            dict(lr_mult=0.005902958103587064, decay_mult=0.0),
            'backbone.blocks.1.attn.k_norm.weight':
            dict(lr_mult=0.005902958103587064, decay_mult=0.0),
            'backbone.blocks.1.attn.gamma.weight':
            dict(lr_mult=0.005902958103587064, decay_mult=0.0),
            'backbone.blocks.1.attn.wq.bias':
            dict(lr_mult=0.005902958103587064, decay_mult=0.0),
            'backbone.blocks.1.attn.wk.bias':
            dict(lr_mult=0.005902958103587064, decay_mult=0.0),
            'backbone.blocks.1.attn.wv.bias':
            dict(lr_mult=0.005902958103587064, decay_mult=0.0),
            'backbone.blocks.1.attn.proj.bias':
            dict(lr_mult=0.005902958103587064, decay_mult=0.0),
            'backbone.blocks.1.ffn.w12.bias':
            dict(lr_mult=0.005902958103587064, decay_mult=0.0),
            'backbone.blocks.1.ffn.w3.bias':
            dict(lr_mult=0.005902958103587064, decay_mult=0.0),
            'backbone.blocks.2.':
            dict(lr_mult=0.00737869762948383),
            'backbone.blocks.2.ln1.weight':
            dict(lr_mult=0.00737869762948383, decay_mult=0.0),
            'backbone.blocks.2.ln2.weight':
            dict(lr_mult=0.00737869762948383, decay_mult=0.0),
            'backbone.blocks.2.attn.q_norm.weight':
            dict(lr_mult=0.00737869762948383, decay_mult=0.0),
            'backbone.blocks.2.attn.k_norm.weight':
            dict(lr_mult=0.00737869762948383, decay_mult=0.0),
            'backbone.blocks.2.attn.gamma.weight':
            dict(lr_mult=0.00737869762948383, decay_mult=0.0),
            'backbone.blocks.2.attn.wq.bias':
            dict(lr_mult=0.00737869762948383, decay_mult=0.0),
            'backbone.blocks.2.attn.wk.bias':
            dict(lr_mult=0.00737869762948383, decay_mult=0.0),
            'backbone.blocks.2.attn.wv.bias':
            dict(lr_mult=0.00737869762948383, decay_mult=0.0),
            'backbone.blocks.2.attn.proj.bias':
            dict(lr_mult=0.00737869762948383, decay_mult=0.0),
            'backbone.blocks.2.ffn.w12.bias':
            dict(lr_mult=0.00737869762948383, decay_mult=0.0),
            'backbone.blocks.2.ffn.w3.bias':
            dict(lr_mult=0.00737869762948383, decay_mult=0.0),
            'backbone.blocks.3.':
            dict(lr_mult=0.009223372036854787),
            'backbone.blocks.3.ln1.weight':
            dict(lr_mult=0.009223372036854787, decay_mult=0.0),
            'backbone.blocks.3.ln2.weight':
            dict(lr_mult=0.009223372036854787, decay_mult=0.0),
            'backbone.blocks.3.attn.q_norm.weight':
            dict(lr_mult=0.009223372036854787, decay_mult=0.0),
            'backbone.blocks.3.attn.k_norm.weight':
            dict(lr_mult=0.009223372036854787, decay_mult=0.0),
            'backbone.blocks.3.attn.gamma.weight':
            dict(lr_mult=0.009223372036854787, decay_mult=0.0),
            'backbone.blocks.3.attn.wq.bias':
            dict(lr_mult=0.009223372036854787, decay_mult=0.0),
            'backbone.blocks.3.attn.wk.bias':
            dict(lr_mult=0.009223372036854787, decay_mult=0.0),
            'backbone.blocks.3.attn.wv.bias':
            dict(lr_mult=0.009223372036854787, decay_mult=0.0),
            'backbone.blocks.3.attn.proj.bias':
            dict(lr_mult=0.009223372036854787, decay_mult=0.0),
            'backbone.blocks.3.ffn.w12.bias':
            dict(lr_mult=0.009223372036854787, decay_mult=0.0),
            'backbone.blocks.3.ffn.w3.bias':
            dict(lr_mult=0.009223372036854787, decay_mult=0.0),
            'backbone.blocks.4.':
            dict(lr_mult=0.011529215046068483),
            'backbone.blocks.4.ln1.weight':
            dict(lr_mult=0.011529215046068483, decay_mult=0.0),
            'backbone.blocks.4.ln2.weight':
            dict(lr_mult=0.011529215046068483, decay_mult=0.0),
            'backbone.blocks.4.attn.q_norm.weight':
            dict(lr_mult=0.011529215046068483, decay_mult=0.0),
            'backbone.blocks.4.attn.k_norm.weight':
            dict(lr_mult=0.011529215046068483, decay_mult=0.0),
            'backbone.blocks.4.attn.gamma.weight':
            dict(lr_mult=0.011529215046068483, decay_mult=0.0),
            'backbone.blocks.4.attn.wq.bias':
            dict(lr_mult=0.011529215046068483, decay_mult=0.0),
            'backbone.blocks.4.attn.wk.bias':
            dict(lr_mult=0.011529215046068483, decay_mult=0.0),
            'backbone.blocks.4.attn.wv.bias':
            dict(lr_mult=0.011529215046068483, decay_mult=0.0),
            'backbone.blocks.4.attn.proj.bias':
            dict(lr_mult=0.011529215046068483, decay_mult=0.0),
            'backbone.blocks.4.ffn.w12.bias':
            dict(lr_mult=0.011529215046068483, decay_mult=0.0),
            'backbone.blocks.4.ffn.w3.bias':
            dict(lr_mult=0.011529215046068483, decay_mult=0.0),
            'backbone.blocks.5.':
            dict(lr_mult=0.014411518807585602),
            'backbone.blocks.5.ln1.weight':
            dict(lr_mult=0.014411518807585602, decay_mult=0.0),
            'backbone.blocks.5.ln2.weight':
            dict(lr_mult=0.014411518807585602, decay_mult=0.0),
            'backbone.blocks.5.attn.q_norm.weight':
            dict(lr_mult=0.014411518807585602, decay_mult=0.0),
            'backbone.blocks.5.attn.k_norm.weight':
            dict(lr_mult=0.014411518807585602, decay_mult=0.0),
            'backbone.blocks.5.attn.gamma.weight':
            dict(lr_mult=0.014411518807585602, decay_mult=0.0),
            'backbone.blocks.5.attn.wq.bias':
            dict(lr_mult=0.014411518807585602, decay_mult=0.0),
            'backbone.blocks.5.attn.wk.bias':
            dict(lr_mult=0.014411518807585602, decay_mult=0.0),
            'backbone.blocks.5.attn.wv.bias':
            dict(lr_mult=0.014411518807585602, decay_mult=0.0),
            'backbone.blocks.5.attn.proj.bias':
            dict(lr_mult=0.014411518807585602, decay_mult=0.0),
            'backbone.blocks.5.ffn.w12.bias':
            dict(lr_mult=0.014411518807585602, decay_mult=0.0),
            'backbone.blocks.5.ffn.w3.bias':
            dict(lr_mult=0.014411518807585602, decay_mult=0.0),
            'backbone.blocks.6.':
            dict(lr_mult=0.018014398509482003),
            'backbone.blocks.6.ln1.weight':
            dict(lr_mult=0.018014398509482003, decay_mult=0.0),
            'backbone.blocks.6.ln2.weight':
            dict(lr_mult=0.018014398509482003, decay_mult=0.0),
            'backbone.blocks.6.attn.q_norm.weight':
            dict(lr_mult=0.018014398509482003, decay_mult=0.0),
            'backbone.blocks.6.attn.k_norm.weight':
            dict(lr_mult=0.018014398509482003, decay_mult=0.0),
            'backbone.blocks.6.attn.gamma.weight':
            dict(lr_mult=0.018014398509482003, decay_mult=0.0),
            'backbone.blocks.6.attn.wq.bias':
            dict(lr_mult=0.018014398509482003, decay_mult=0.0),
            'backbone.blocks.6.attn.wk.bias':
            dict(lr_mult=0.018014398509482003, decay_mult=0.0),
            'backbone.blocks.6.attn.wv.bias':
            dict(lr_mult=0.018014398509482003, decay_mult=0.0),
            'backbone.blocks.6.attn.proj.bias':
            dict(lr_mult=0.018014398509482003, decay_mult=0.0),
            'backbone.blocks.6.ffn.w12.bias':
            dict(lr_mult=0.018014398509482003, decay_mult=0.0),
            'backbone.blocks.6.ffn.w3.bias':
            dict(lr_mult=0.018014398509482003, decay_mult=0.0),
            'backbone.blocks.7.':
            dict(lr_mult=0.022517998136852502),
            'backbone.blocks.7.ln1.weight':
            dict(lr_mult=0.022517998136852502, decay_mult=0.0),
            'backbone.blocks.7.ln2.weight':
            dict(lr_mult=0.022517998136852502, decay_mult=0.0),
            'backbone.blocks.7.attn.q_norm.weight':
            dict(lr_mult=0.022517998136852502, decay_mult=0.0),
            'backbone.blocks.7.attn.k_norm.weight':
            dict(lr_mult=0.022517998136852502, decay_mult=0.0),
            'backbone.blocks.7.attn.gamma.weight':
            dict(lr_mult=0.022517998136852502, decay_mult=0.0),
            'backbone.blocks.7.attn.wq.bias':
            dict(lr_mult=0.022517998136852502, decay_mult=0.0),
            'backbone.blocks.7.attn.wk.bias':
            dict(lr_mult=0.022517998136852502, decay_mult=0.0),
            'backbone.blocks.7.attn.wv.bias':
            dict(lr_mult=0.022517998136852502, decay_mult=0.0),
            'backbone.blocks.7.attn.proj.bias':
            dict(lr_mult=0.022517998136852502, decay_mult=0.0),
            'backbone.blocks.7.ffn.w12.bias':
            dict(lr_mult=0.022517998136852502, decay_mult=0.0),
            'backbone.blocks.7.ffn.w3.bias':
            dict(lr_mult=0.022517998136852502, decay_mult=0.0),
            'backbone.blocks.8.':
            dict(lr_mult=0.028147497671065624),
            'backbone.blocks.8.ln1.weight':
            dict(lr_mult=0.028147497671065624, decay_mult=0.0),
            'backbone.blocks.8.ln2.weight':
            dict(lr_mult=0.028147497671065624, decay_mult=0.0),
            'backbone.blocks.8.attn.q_norm.weight':
            dict(lr_mult=0.028147497671065624, decay_mult=0.0),
            'backbone.blocks.8.attn.k_norm.weight':
            dict(lr_mult=0.028147497671065624, decay_mult=0.0),
            'backbone.blocks.8.attn.gamma.weight':
            dict(lr_mult=0.028147497671065624, decay_mult=0.0),
            'backbone.blocks.8.attn.wq.bias':
            dict(lr_mult=0.028147497671065624, decay_mult=0.0),
            'backbone.blocks.8.attn.wk.bias':
            dict(lr_mult=0.028147497671065624, decay_mult=0.0),
            'backbone.blocks.8.attn.wv.bias':
            dict(lr_mult=0.028147497671065624, decay_mult=0.0),
            'backbone.blocks.8.attn.proj.bias':
            dict(lr_mult=0.028147497671065624, decay_mult=0.0),
            'backbone.blocks.8.ffn.w12.bias':
            dict(lr_mult=0.028147497671065624, decay_mult=0.0),
            'backbone.blocks.8.ffn.w3.bias':
            dict(lr_mult=0.028147497671065624, decay_mult=0.0),
            'backbone.blocks.9.':
            dict(lr_mult=0.03518437208883203),
            'backbone.blocks.9.ln1.weight':
            dict(lr_mult=0.03518437208883203, decay_mult=0.0),
            'backbone.blocks.9.ln2.weight':
            dict(lr_mult=0.03518437208883203, decay_mult=0.0),
            'backbone.blocks.9.attn.q_norm.weight':
            dict(lr_mult=0.03518437208883203, decay_mult=0.0),
            'backbone.blocks.9.attn.k_norm.weight':
            dict(lr_mult=0.03518437208883203, decay_mult=0.0),
            'backbone.blocks.9.attn.gamma.weight':
            dict(lr_mult=0.03518437208883203, decay_mult=0.0),
            'backbone.blocks.9.attn.wq.bias':
            dict(lr_mult=0.03518437208883203, decay_mult=0.0),
            'backbone.blocks.9.attn.wk.bias':
            dict(lr_mult=0.03518437208883203, decay_mult=0.0),
            'backbone.blocks.9.attn.wv.bias':
            dict(lr_mult=0.03518437208883203, decay_mult=0.0),
            'backbone.blocks.9.attn.proj.bias':
            dict(lr_mult=0.03518437208883203, decay_mult=0.0),
            'backbone.blocks.9.ffn.w12.bias':
            dict(lr_mult=0.03518437208883203, decay_mult=0.0),
            'backbone.blocks.9.ffn.w3.bias':
            dict(lr_mult=0.03518437208883203, decay_mult=0.0),
            'backbone.blocks.10.':
            dict(lr_mult=0.043980465111040035),
            'backbone.blocks.10.ln1.weight':
            dict(lr_mult=0.043980465111040035, decay_mult=0.0),
            'backbone.blocks.10.ln2.weight':
            dict(lr_mult=0.043980465111040035, decay_mult=0.0),
            'backbone.blocks.10.attn.q_norm.weight':
            dict(lr_mult=0.043980465111040035, decay_mult=0.0),
            'backbone.blocks.10.attn.k_norm.weight':
            dict(lr_mult=0.043980465111040035, decay_mult=0.0),
            'backbone.blocks.10.attn.gamma.weight':
            dict(lr_mult=0.043980465111040035, decay_mult=0.0),
            'backbone.blocks.10.attn.wq.bias':
            dict(lr_mult=0.043980465111040035, decay_mult=0.0),
            'backbone.blocks.10.attn.wk.bias':
            dict(lr_mult=0.043980465111040035, decay_mult=0.0),
            'backbone.blocks.10.attn.wv.bias':
            dict(lr_mult=0.043980465111040035, decay_mult=0.0),
            'backbone.blocks.10.attn.proj.bias':
            dict(lr_mult=0.043980465111040035, decay_mult=0.0),
            'backbone.blocks.10.ffn.w12.bias':
            dict(lr_mult=0.043980465111040035, decay_mult=0.0),
            'backbone.blocks.10.ffn.w3.bias':
            dict(lr_mult=0.043980465111040035, decay_mult=0.0),
            'backbone.blocks.11.':
            dict(lr_mult=0.054975581388800036),
            'backbone.blocks.11.ln1.weight':
            dict(lr_mult=0.054975581388800036, decay_mult=0.0),
            'backbone.blocks.11.ln2.weight':
            dict(lr_mult=0.054975581388800036, decay_mult=0.0),
            'backbone.blocks.11.attn.q_norm.weight':
            dict(lr_mult=0.054975581388800036, decay_mult=0.0),
            'backbone.blocks.11.attn.k_norm.weight':
            dict(lr_mult=0.054975581388800036, decay_mult=0.0),
            'backbone.blocks.11.attn.gamma.weight':
            dict(lr_mult=0.054975581388800036, decay_mult=0.0),
            'backbone.blocks.11.attn.wq.bias':
            dict(lr_mult=0.054975581388800036, decay_mult=0.0),
            'backbone.blocks.11.attn.wk.bias':
            dict(lr_mult=0.054975581388800036, decay_mult=0.0),
            'backbone.blocks.11.attn.wv.bias':
            dict(lr_mult=0.054975581388800036, decay_mult=0.0),
            'backbone.blocks.11.attn.proj.bias':
            dict(lr_mult=0.054975581388800036, decay_mult=0.0),
            'backbone.blocks.11.ffn.w12.bias':
            dict(lr_mult=0.054975581388800036, decay_mult=0.0),
            'backbone.blocks.11.ffn.w3.bias':
            dict(lr_mult=0.054975581388800036, decay_mult=0.0),
            'backbone.blocks.12.':
            dict(lr_mult=0.06871947673600004),
            'backbone.blocks.12.ln1.weight':
            dict(lr_mult=0.06871947673600004, decay_mult=0.0),
            'backbone.blocks.12.ln2.weight':
            dict(lr_mult=0.06871947673600004, decay_mult=0.0),
            'backbone.blocks.12.attn.q_norm.weight':
            dict(lr_mult=0.06871947673600004, decay_mult=0.0),
            'backbone.blocks.12.attn.k_norm.weight':
            dict(lr_mult=0.06871947673600004, decay_mult=0.0),
            'backbone.blocks.12.attn.gamma.weight':
            dict(lr_mult=0.06871947673600004, decay_mult=0.0),
            'backbone.blocks.12.attn.wq.bias':
            dict(lr_mult=0.06871947673600004, decay_mult=0.0),
            'backbone.blocks.12.attn.wk.bias':
            dict(lr_mult=0.06871947673600004, decay_mult=0.0),
            'backbone.blocks.12.attn.wv.bias':
            dict(lr_mult=0.06871947673600004, decay_mult=0.0),
            'backbone.blocks.12.attn.proj.bias':
            dict(lr_mult=0.06871947673600004, decay_mult=0.0),
            'backbone.blocks.12.ffn.w12.bias':
            dict(lr_mult=0.06871947673600004, decay_mult=0.0),
            'backbone.blocks.12.ffn.w3.bias':
            dict(lr_mult=0.06871947673600004, decay_mult=0.0),
            'backbone.blocks.13.':
            dict(lr_mult=0.08589934592000005),
            'backbone.blocks.13.ln1.weight':
            dict(lr_mult=0.08589934592000005, decay_mult=0.0),
            'backbone.blocks.13.ln2.weight':
            dict(lr_mult=0.08589934592000005, decay_mult=0.0),
            'backbone.blocks.13.attn.q_norm.weight':
            dict(lr_mult=0.08589934592000005, decay_mult=0.0),
            'backbone.blocks.13.attn.k_norm.weight':
            dict(lr_mult=0.08589934592000005, decay_mult=0.0),
            'backbone.blocks.13.attn.gamma.weight':
            dict(lr_mult=0.08589934592000005, decay_mult=0.0),
            'backbone.blocks.13.attn.wq.bias':
            dict(lr_mult=0.08589934592000005, decay_mult=0.0),
            'backbone.blocks.13.attn.wk.bias':
            dict(lr_mult=0.08589934592000005, decay_mult=0.0),
            'backbone.blocks.13.attn.wv.bias':
            dict(lr_mult=0.08589934592000005, decay_mult=0.0),
            'backbone.blocks.13.attn.proj.bias':
            dict(lr_mult=0.08589934592000005, decay_mult=0.0),
            'backbone.blocks.13.ffn.w12.bias':
            dict(lr_mult=0.08589934592000005, decay_mult=0.0),
            'backbone.blocks.13.ffn.w3.bias':
            dict(lr_mult=0.08589934592000005, decay_mult=0.0),
            'backbone.blocks.14.':
            dict(lr_mult=0.10737418240000006),
            'backbone.blocks.14.ln1.weight':
            dict(lr_mult=0.10737418240000006, decay_mult=0.0),
            'backbone.blocks.14.ln2.weight':
            dict(lr_mult=0.10737418240000006, decay_mult=0.0),
            'backbone.blocks.14.attn.q_norm.weight':
            dict(lr_mult=0.10737418240000006, decay_mult=0.0),
            'backbone.blocks.14.attn.k_norm.weight':
            dict(lr_mult=0.10737418240000006, decay_mult=0.0),
            'backbone.blocks.14.attn.gamma.weight':
            dict(lr_mult=0.10737418240000006, decay_mult=0.0),
            'backbone.blocks.14.attn.wq.bias':
            dict(lr_mult=0.10737418240000006, decay_mult=0.0),
            'backbone.blocks.14.attn.wk.bias':
            dict(lr_mult=0.10737418240000006, decay_mult=0.0),
            'backbone.blocks.14.attn.wv.bias':
            dict(lr_mult=0.10737418240000006, decay_mult=0.0),
            'backbone.blocks.14.attn.proj.bias':
            dict(lr_mult=0.10737418240000006, decay_mult=0.0),
            'backbone.blocks.14.ffn.w12.bias':
            dict(lr_mult=0.10737418240000006, decay_mult=0.0),
            'backbone.blocks.14.ffn.w3.bias':
            dict(lr_mult=0.10737418240000006, decay_mult=0.0),
            'backbone.blocks.15.':
            dict(lr_mult=0.13421772800000006),
            'backbone.blocks.15.ln1.weight':
            dict(lr_mult=0.13421772800000006, decay_mult=0.0),
            'backbone.blocks.15.ln2.weight':
            dict(lr_mult=0.13421772800000006, decay_mult=0.0),
            'backbone.blocks.15.attn.q_norm.weight':
            dict(lr_mult=0.13421772800000006, decay_mult=0.0),
            'backbone.blocks.15.attn.k_norm.weight':
            dict(lr_mult=0.13421772800000006, decay_mult=0.0),
            'backbone.blocks.15.attn.gamma.weight':
            dict(lr_mult=0.13421772800000006, decay_mult=0.0),
            'backbone.blocks.15.attn.wq.bias':
            dict(lr_mult=0.13421772800000006, decay_mult=0.0),
            'backbone.blocks.15.attn.wk.bias':
            dict(lr_mult=0.13421772800000006, decay_mult=0.0),
            'backbone.blocks.15.attn.wv.bias':
            dict(lr_mult=0.13421772800000006, decay_mult=0.0),
            'backbone.blocks.15.attn.proj.bias':
            dict(lr_mult=0.13421772800000006, decay_mult=0.0),
            'backbone.blocks.15.ffn.w12.bias':
            dict(lr_mult=0.13421772800000006, decay_mult=0.0),
            'backbone.blocks.15.ffn.w3.bias':
            dict(lr_mult=0.13421772800000006, decay_mult=0.0),
            'backbone.blocks.16.':
            dict(lr_mult=0.1677721600000001),
            'backbone.blocks.16.ln1.weight':
            dict(lr_mult=0.1677721600000001, decay_mult=0.0),
            'backbone.blocks.16.ln2.weight':
            dict(lr_mult=0.1677721600000001, decay_mult=0.0),
            'backbone.blocks.16.attn.q_norm.weight':
            dict(lr_mult=0.1677721600000001, decay_mult=0.0),
            'backbone.blocks.16.attn.k_norm.weight':
            dict(lr_mult=0.1677721600000001, decay_mult=0.0),
            'backbone.blocks.16.attn.gamma.weight':
            dict(lr_mult=0.1677721600000001, decay_mult=0.0),
            'backbone.blocks.16.attn.wq.bias':
            dict(lr_mult=0.1677721600000001, decay_mult=0.0),
            'backbone.blocks.16.attn.wk.bias':
            dict(lr_mult=0.1677721600000001, decay_mult=0.0),
            'backbone.blocks.16.attn.wv.bias':
            dict(lr_mult=0.1677721600000001, decay_mult=0.0),
            'backbone.blocks.16.attn.proj.bias':
            dict(lr_mult=0.1677721600000001, decay_mult=0.0),
            'backbone.blocks.16.ffn.w12.bias':
            dict(lr_mult=0.1677721600000001, decay_mult=0.0),
            'backbone.blocks.16.ffn.w3.bias':
            dict(lr_mult=0.1677721600000001, decay_mult=0.0),
            'backbone.blocks.17.':
            dict(lr_mult=0.20971520000000007),
            'backbone.blocks.17.ln1.weight':
            dict(lr_mult=0.20971520000000007, decay_mult=0.0),
            'backbone.blocks.17.ln2.weight':
            dict(lr_mult=0.20971520000000007, decay_mult=0.0),
            'backbone.blocks.17.attn.q_norm.weight':
            dict(lr_mult=0.20971520000000007, decay_mult=0.0),
            'backbone.blocks.17.attn.k_norm.weight':
            dict(lr_mult=0.20971520000000007, decay_mult=0.0),
            'backbone.blocks.17.attn.gamma.weight':
            dict(lr_mult=0.20971520000000007, decay_mult=0.0),
            'backbone.blocks.17.attn.wq.bias':
            dict(lr_mult=0.20971520000000007, decay_mult=0.0),
            'backbone.blocks.17.attn.wk.bias':
            dict(lr_mult=0.20971520000000007, decay_mult=0.0),
            'backbone.blocks.17.attn.wv.bias':
            dict(lr_mult=0.20971520000000007, decay_mult=0.0),
            'backbone.blocks.17.attn.proj.bias':
            dict(lr_mult=0.20971520000000007, decay_mult=0.0),
            'backbone.blocks.17.ffn.w12.bias':
            dict(lr_mult=0.20971520000000007, decay_mult=0.0),
            'backbone.blocks.17.ffn.w3.bias':
            dict(lr_mult=0.20971520000000007, decay_mult=0.0),
            'backbone.blocks.18.':
            dict(lr_mult=0.2621440000000001),
            'backbone.blocks.18.ln1.weight':
            dict(lr_mult=0.2621440000000001, decay_mult=0.0),
            'backbone.blocks.18.ln2.weight':
            dict(lr_mult=0.2621440000000001, decay_mult=0.0),
            'backbone.blocks.18.attn.q_norm.weight':
            dict(lr_mult=0.2621440000000001, decay_mult=0.0),
            'backbone.blocks.18.attn.k_norm.weight':
            dict(lr_mult=0.2621440000000001, decay_mult=0.0),
            'backbone.blocks.18.attn.gamma.weight':
            dict(lr_mult=0.2621440000000001, decay_mult=0.0),
            'backbone.blocks.18.attn.wq.bias':
            dict(lr_mult=0.2621440000000001, decay_mult=0.0),
            'backbone.blocks.18.attn.wk.bias':
            dict(lr_mult=0.2621440000000001, decay_mult=0.0),
            'backbone.blocks.18.attn.wv.bias':
            dict(lr_mult=0.2621440000000001, decay_mult=0.0),
            'backbone.blocks.18.attn.proj.bias':
            dict(lr_mult=0.2621440000000001, decay_mult=0.0),
            'backbone.blocks.18.ffn.w12.bias':
            dict(lr_mult=0.2621440000000001, decay_mult=0.0),
            'backbone.blocks.18.ffn.w3.bias':
            dict(lr_mult=0.2621440000000001, decay_mult=0.0),
            'backbone.blocks.19.':
            dict(lr_mult=0.3276800000000001),
            'backbone.blocks.19.ln1.weight':
            dict(lr_mult=0.3276800000000001, decay_mult=0.0),
            'backbone.blocks.19.ln2.weight':
            dict(lr_mult=0.3276800000000001, decay_mult=0.0),
            'backbone.blocks.19.attn.q_norm.weight':
            dict(lr_mult=0.3276800000000001, decay_mult=0.0),
            'backbone.blocks.19.attn.k_norm.weight':
            dict(lr_mult=0.3276800000000001, decay_mult=0.0),
            'backbone.blocks.19.attn.gamma.weight':
            dict(lr_mult=0.3276800000000001, decay_mult=0.0),
            'backbone.blocks.19.attn.wq.bias':
            dict(lr_mult=0.3276800000000001, decay_mult=0.0),
            'backbone.blocks.19.attn.wk.bias':
            dict(lr_mult=0.3276800000000001, decay_mult=0.0),
            'backbone.blocks.19.attn.wv.bias':
            dict(lr_mult=0.3276800000000001, decay_mult=0.0),
            'backbone.blocks.19.attn.proj.bias':
            dict(lr_mult=0.3276800000000001, decay_mult=0.0),
            'backbone.blocks.19.ffn.w12.bias':
            dict(lr_mult=0.3276800000000001, decay_mult=0.0),
            'backbone.blocks.19.ffn.w3.bias':
            dict(lr_mult=0.3276800000000001, decay_mult=0.0),
            'backbone.blocks.20.':
            dict(lr_mult=0.4096000000000001),
            'backbone.blocks.20.ln1.weight':
            dict(lr_mult=0.4096000000000001, decay_mult=0.0),
            'backbone.blocks.20.ln2.weight':
            dict(lr_mult=0.4096000000000001, decay_mult=0.0),
            'backbone.blocks.20.attn.q_norm.weight':
            dict(lr_mult=0.4096000000000001, decay_mult=0.0),
            'backbone.blocks.20.attn.k_norm.weight':
            dict(lr_mult=0.4096000000000001, decay_mult=0.0),
            'backbone.blocks.20.attn.gamma.weight':
            dict(lr_mult=0.4096000000000001, decay_mult=0.0),
            'backbone.blocks.20.attn.wq.bias':
            dict(lr_mult=0.4096000000000001, decay_mult=0.0),
            'backbone.blocks.20.attn.wk.bias':
            dict(lr_mult=0.4096000000000001, decay_mult=0.0),
            'backbone.blocks.20.attn.wv.bias':
            dict(lr_mult=0.4096000000000001, decay_mult=0.0),
            'backbone.blocks.20.attn.proj.bias':
            dict(lr_mult=0.4096000000000001, decay_mult=0.0),
            'backbone.blocks.20.ffn.w12.bias':
            dict(lr_mult=0.4096000000000001, decay_mult=0.0),
            'backbone.blocks.20.ffn.w3.bias':
            dict(lr_mult=0.4096000000000001, decay_mult=0.0),
            'backbone.blocks.21.':
            dict(lr_mult=0.5120000000000001),
            'backbone.blocks.21.ln1.weight':
            dict(lr_mult=0.5120000000000001, decay_mult=0.0),
            'backbone.blocks.21.ln2.weight':
            dict(lr_mult=0.5120000000000001, decay_mult=0.0),
            'backbone.blocks.21.attn.q_norm.weight':
            dict(lr_mult=0.5120000000000001, decay_mult=0.0),
            'backbone.blocks.21.attn.k_norm.weight':
            dict(lr_mult=0.5120000000000001, decay_mult=0.0),
            'backbone.blocks.21.attn.gamma.weight':
            dict(lr_mult=0.5120000000000001, decay_mult=0.0),
            'backbone.blocks.21.attn.wq.bias':
            dict(lr_mult=0.5120000000000001, decay_mult=0.0),
            'backbone.blocks.21.attn.wk.bias':
            dict(lr_mult=0.5120000000000001, decay_mult=0.0),
            'backbone.blocks.21.attn.wv.bias':
            dict(lr_mult=0.5120000000000001, decay_mult=0.0),
            'backbone.blocks.21.attn.proj.bias':
            dict(lr_mult=0.5120000000000001, decay_mult=0.0),
            'backbone.blocks.21.ffn.w12.bias':
            dict(lr_mult=0.5120000000000001, decay_mult=0.0),
            'backbone.blocks.21.ffn.w3.bias':
            dict(lr_mult=0.5120000000000001, decay_mult=0.0),
            'backbone.blocks.22.':
            dict(lr_mult=0.6400000000000001),
            'backbone.blocks.22.ln1.weight':
            dict(lr_mult=0.6400000000000001, decay_mult=0.0),
            'backbone.blocks.22.ln2.weight':
            dict(lr_mult=0.6400000000000001, decay_mult=0.0),
            'backbone.blocks.22.attn.q_norm.weight':
            dict(lr_mult=0.6400000000000001, decay_mult=0.0),
            'backbone.blocks.22.attn.k_norm.weight':
            dict(lr_mult=0.6400000000000001, decay_mult=0.0),
            'backbone.blocks.22.attn.gamma.weight':
            dict(lr_mult=0.6400000000000001, decay_mult=0.0),
            'backbone.blocks.22.attn.wq.bias':
            dict(lr_mult=0.6400000000000001, decay_mult=0.0),
            'backbone.blocks.22.attn.wk.bias':
            dict(lr_mult=0.6400000000000001, decay_mult=0.0),
            'backbone.blocks.22.attn.wv.bias':
            dict(lr_mult=0.6400000000000001, decay_mult=0.0),
            'backbone.blocks.22.attn.proj.bias':
            dict(lr_mult=0.6400000000000001, decay_mult=0.0),
            'backbone.blocks.22.ffn.w12.bias':
            dict(lr_mult=0.6400000000000001, decay_mult=0.0),
            'backbone.blocks.22.ffn.w3.bias':
            dict(lr_mult=0.6400000000000001, decay_mult=0.0),
            'backbone.blocks.23.':
            dict(lr_mult=0.8),
            'backbone.blocks.23.ln1.weight':
            dict(lr_mult=0.8, decay_mult=0.0),
            'backbone.blocks.23.ln2.weight':
            dict(lr_mult=0.8, decay_mult=0.0),
            'backbone.blocks.23.attn.q_norm.weight':
            dict(lr_mult=0.8, decay_mult=0.0),
            'backbone.blocks.23.attn.k_norm.weight':
            dict(lr_mult=0.8, decay_mult=0.0),
            'backbone.blocks.23.attn.gamma.weight':
            dict(lr_mult=0.8, decay_mult=0.0),
            'backbone.blocks.23.attn.wq.bias':
            dict(lr_mult=0.8, decay_mult=0.0),
            'backbone.blocks.23.attn.wk.bias':
            dict(lr_mult=0.8, decay_mult=0.0),
            'backbone.blocks.23.attn.wv.bias':
            dict(lr_mult=0.8, decay_mult=0.0),
            'backbone.blocks.23.attn.proj.bias':
            dict(lr_mult=0.8, decay_mult=0.0),
            'backbone.blocks.23.ffn.w12.bias':
            dict(lr_mult=0.8, decay_mult=0.0),
            'backbone.blocks.23.ffn.w3.bias':
            dict(lr_mult=0.8, decay_mult=0.0),
            'backbone.ln1.':
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
    step=[14, 18])
total_epochs = 20
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
work_dir = '/mnt/cachefs/esteban/pii/runs/train/wd_armAS_fluence'
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
        type='RandomRectCrop',
        crop_choice=[0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        aspect=1.3333333333333333,
        bbox_clip_border=False),
    dict(
        type='Resize',
        img_scale=(1024, 768),
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
        img_scale=(1024, 768),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[128.0, 128.0, 128.0],
                to_rgb=True),
            dict(type='Pad', size=(768, 1024), pad_val=128),
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
                type='RandomRectCrop',
                crop_choice=[
                    0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0
                ],
                aspect=1.3333333333333333,
                bbox_clip_border=False),
            dict(
                type='Resize',
                img_scale=(1024, 768),
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
                img_scale=(1024, 768),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(type='RandomFlip', flip_ratio=0.0),
                    dict(
                        type='Normalize',
                        mean=[127.5, 127.5, 127.5],
                        std=[128.0, 128.0, 128.0],
                        to_rgb=True),
                    dict(type='Pad', size=(768, 1024), pad_val=128),
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
                img_scale=(1024, 768),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(type='RandomFlip', flip_ratio=0.0),
                    dict(
                        type='Normalize',
                        mean=[127.5, 127.5, 127.5],
                        std=[128.0, 128.0, 128.0],
                        to_rgb=True),
                    dict(type='Pad', size=(768, 1024), pad_val=128),
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
        type='Sapiens2ViT',
        arch='sapiens2_0.4b',
        patch_size=16,
        img_size=(768, 1024),
        out_indices=(23, ),
        drop_path_rate=0.3,
        pretrained=
        '/mnt/cachefs/esteban/pii/weights/sapiens2_0.4b_pretrain.safetensors',
        bf16=True,
        use_checkpoint=False),
    neck=dict(
        type='SimpleFPN16',
        in_channels=1024,
        out_channels=128,
        num_inputs=1,
        fuse_inputs=False,
        up_mode='deconv',
        down_mode='avg'),
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
    imports=['sapiens2_backbone', 'simple_fpn16', 'rect_crop'],
    allow_failed_imports=False)
gpu_ids = range(0, 8)
