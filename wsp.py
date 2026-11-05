"""Weight-space edit toolkit on cached features. Edits act on the two projection matrices:
   image emb = normalize(pooled_v @ Wv.T), text emb = normalize(pooled_t @ Wt.T)  (Wv:[512,768], Wt:[512,512])
Every edit is a function (Wv, Wt) -> (Wv', Wt'); accuracy is scored on the SEARCH split (train images) only,
and the full ImageNet val split is touched only for the final report."""
import torch, torch.nn.functional as F
dev = 'cuda'
P = torch.load('proj.pt'); Wv0, Wt0 = P['Wv'].to(dev).float(), P['Wt'].to(dev).float()
TXT = torch.load('cache_text.pt').to(dev)              # [1000,80,512]
def load(split):
    f, y = torch.load(f'cache_{split}.pt'); return f.to(dev).float(), y.to(dev)
def classifier(Wt):
    e = F.normalize(TXT @ Wt.T, dim=-1).mean(1); return F.normalize(e, dim=-1)
def acc(Wv, Wt, data):
    f, y = data; z = F.normalize(f @ Wv.T, dim=-1)
    return ((z @ classifier(Wt).T).argmax(1) == y).float().mean().item() * 100
