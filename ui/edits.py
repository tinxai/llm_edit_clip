"""Edit ops on CLIP ViT-B/32 vision tower. head h of layer l = out_proj columns [h*64:(h+1)*64]; MLP block = fc1/fc2."""
import torch
def apply_ops(vm, orig, ops):
    with torch.no_grad():
        for n, p in vm.named_parameters(): p.copy_(orig[n])
        for o in ops:
            L = vm.encoder.layers[int(o['layer'])]; s = 0.0 if o.get('mode', 'zero') == 'zero' else float(o.get('value', 1.0))
            if o['type'] == 'head':
                h = int(o['head']); L.self_attn.out_proj.weight[:, h*64:(h+1)*64] *= s
            elif o['type'] == 'mlp':
                L.mlp.fc2.weight.mul_(s); L.mlp.fc2.bias.mul_(s)
            else: raise ValueError(o)
