import sys, json, torch, torch.nn.functional as F, numpy as np
import os
import harness as H, typo, zs_meta
from wsp import classifier, Wt0
sys.path.insert(0, 'ui'); from edits import apply_ops
from PIL import Image
m = H.load_model(); vm, proj = m.vision_model, m.visual_projection; Wc = classifier(Wt0)
orig = {n: p.detach().clone() for n, p in vm.named_parameters()}
C, A, Y = torch.load('typo/TA.pt'); N = len(Y)
typed = torch.tensor([zs_meta.IMAGENET_CLASSNAMES.index(typo.wrong_name(int(Y[i]), 100000 + i)) for i in range(N)])
idx = torch.arange(0, N, 4); Cs, As, Ys = C[idx].cuda(), A[idx].cuda(), Y[idx].cuda()
@torch.no_grad()
def acc(X, Yl):
    z = []
    with torch.autocast('cuda', dtype=torch.float16):
        for i in range(0, len(X), 500): z.append(F.normalize(proj(vm(pixel_values=X[i:i+500]).pooler_output).float(), dim=-1))
    return ((torch.cat(z) @ Wc.T).argmax(1) == Yl).float().mean().item() * 100
apply_ops(vm, orig, []); base = (acc(As, Ys), acc(Cs, Ys)); print('base', base, flush=True)
scan = {}
for l in range(12):
    for h in range(12):
        apply_ops(vm, orig, [{'type': 'head', 'layer': l, 'head': h}]); scan[f'H{l}.{h}'] = [acc(As, Ys), acc(Cs, Ys)]
    apply_ops(vm, orig, [{'type': 'mlp', 'layer': l}]); scan[f'M{l}'] = [acc(As, Ys), acc(Cs, Ys)]
    print('layer', l, flush=True)
apply_ops(vm, orig, [])
# ---- readouts: direct effect of each head's CLS output on every class score (linearised through final LN + projection)
cap = {}
def pre(l):
    def f(mod, args): cap[('z', l)] = args[0].detach()
    return f
def ln1(l):
    def f(mod, args, out): cap[('ln', l)] = out.detach()
    return f
for l, L in enumerate(vm.encoder.layers):
    L.self_attn.out_proj.register_forward_pre_hook(pre(l)); L.layer_norm1.register_forward_hook(ln1(l))
g = vm.post_layernorm.weight
@torch.no_grad()
def run(X):
    out = vm(pixel_values=X, output_hidden_states=True); hs = out.hidden_states[-1][:, 0].float()
    std = (hs - hs.mean(-1, keepdim=True)).pow(2).mean(-1, keepdim=True).add(1e-5).sqrt()
    dla = torch.zeros(len(X), 12, 12, 1000, device='cuda')
    attn = torch.zeros(len(X), 12, 12, 49, device='cuda')
    for l, L in enumerate(vm.encoder.layers):
        z = cap[('z', l)][:, 0].float().view(len(X), 12, 64); Wo = L.self_attn.out_proj.weight.float().view(768, 12, 64)
        c = torch.einsum('nhd,ohd->nho', z, Wo)  # [n,12,768] head contributions
        c = (c - c.mean(-1, keepdim=True)) / std[:, None] * g.float()
        e = c @ proj.weight.float().T  # [n,12,512]
        dla[:, l] = e @ Wc.T
        x = cap[('ln', l)].float(); at = L.self_attn
        q = at.q_proj(x[:, :1]).view(len(X), 1, 12, 64).transpose(1, 2); k = at.k_proj(x).view(len(X), 50, 12, 64).transpose(1, 2)
        a = ((q @ k.transpose(-1, -2)) * 64 ** -0.5).softmax(-1)[:, :, 0]  # [n,12,50]
        attn[:, l] = a[:, :, 1:]
    return dla, attn
sel = torch.arange(0, N, 8)[:400]
read = torch.zeros(12, 12); readc = torch.zeros(12, 12)
with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
    for i in range(0, len(sel), 100):
        j = sel[i:i+100]; t = typed[j].cuda(); y = Y[j].cuda()
        for X, tgt in [(A[j].cuda(), read), (C[j].cuda(), readc)]:
            d, _ = run(X); d = d.float()
            tt = d.gather(3, t.view(-1, 1, 1, 1).expand(-1, 12, 12, 1)).squeeze(-1)  # typed-word score
            tm = d.mean(-1)
            tgt += ((tt - tm).sum(0)).cpu() / len(sel)
readout = (read - readc)
# ---- example images
ex_ids = [101, 801, 1201, 1801, 2601, 3401]; ex = []
mean = torch.tensor(H.MEAN).view(3, 1, 1); std_ = torch.tensor(H.STD).view(3, 1, 1)
def png(t, path): Image.fromarray(((t.float() * std_ + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(path)
with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
    for k, i in enumerate(ex_ids):
        png(C[i], f'ui/img/ex{k}_clean.png'); png(A[i], f'ui/img/ex{k}_att.png')
        d, at = run(A[i:i+1].cuda()); d = d.float()[0]; at = at.float()[0]
        words = {}
        for l in range(12):
            for h in range(12):
                top = d[l, h].topk(4).indices.tolist(); words[f'{l}.{h}'] = [zs_meta.IMAGENET_CLASSNAMES[t] for t in top]
        ex.append({'true': zs_meta.IMAGENET_CLASSNAMES[int(Y[i])], 'typed': zs_meta.IMAGENET_CLASSNAMES[int(typed[i])], 'attn': np.round(at.cpu().numpy(), 3).tolist(),
                   'typed_score': np.round((d[:, :, int(typed[i])] - d.mean(-1)).cpu().numpy(), 3).tolist(), 'words': words})
json.dump({'base': base, 'scan': scan, 'readout': np.round(readout.numpy(), 4).tolist(), 'examples': ex}, open('ui/data.json', 'w'))
print('done')
