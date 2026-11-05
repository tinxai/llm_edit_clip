from wsp import *
S, V = load('search'), load('val')
print('baseline search %.2f  val %.2f' % (acc(Wv0,Wt0,S), acc(Wv0,Wt0,V)))
f = S[0]
print('pooled feat: mean-abs top dims', f.abs().mean(0).topk(8))
print('pooled feat std top dims', f.std(0).topk(5))
# single-dim ablation of the 768 residual dims (zero column of Wv) on search split
base = acc(Wv0,Wt0,S); res=[]
for d in range(768):
    W = Wv0.clone(); W[:,d]=0; res.append(acc(W,Wt0,S)-base)
r = torch.tensor(res); print('ablate-dim delta: best', r.topk(8), 'worst', (-r).topk(3))
# modality gap direction
z = F.normalize(S[0] @ Wv0.T, dim=-1); t = classifier(Wt0)
gap = z.mean(0) - t.mean(0); print('gap norm', gap.norm().item(), 'cos(mean img, mean txt)', F.cosine_similarity(z.mean(0),t.mean(0),dim=0).item())
