from wsp import *
S = load('search'); f,y = S
A = (f[[i for i in range(10000) if (i%10)<5]], y[[i for i in range(10000) if (i%10)<5]])
B = (f[[i for i in range(10000) if (i%10)>=5]], y[[i for i in range(10000) if (i%10)>=5]])
def rep(tag, Wv, Wt):
    a,b = acc(Wv,Wt,A), acc(Wv,Wt,B); print('%-34s A %.2f (%+.2f)  B %.2f (%+.2f)'%(tag,a,a-a0,b,b-b0)); return a-a0,b-b0
a0,b0 = acc(Wv0,Wt0,A), acc(Wv0,Wt0,B); print('base A %.2f B %.2f'%(a0,b0))
I = torch.eye(512,device=dev)
zi = F.normalize(f@Wv0.T,dim=-1); tc = classifier(Wt0)
# 1. project modality-gap direction out of image side / text side / both
u = F.normalize(zi.mean(0)-tc.mean(0),dim=0)
for al in [0.25,0.5,1.0]:
    P = I-al*torch.outer(u,u); rep(f'proj gap dir out (img) a={al}', P@Wv0, Wt0); rep(f'proj gap dir out (txt) a={al}', Wv0, P@Wt0); rep(f'proj gap dir out (both) a={al}', P@Wv0, P@Wt0)
# 2. project out mean image direction on image side only
m = F.normalize(zi.mean(0),dim=0)
for al in [0.5,1.0]: rep(f'proj mean-img dir out a={al}', (I-al*torch.outer(m,m))@Wv0, Wt0)
# 3. rescale outlier channel 336
for s in [0.0,0.5,0.8,1.2]:
    W=Wv0.clone(); W[:,336]*=s; rep(f'scale resid ch336 x{s}', W, Wt0)
# 4. low-rank denoise Wv via PCA of pooled features
fc = f - f.mean(0); U,Sg,Vh = torch.linalg.svd(fc, full_matrices=False)
for k in [128,256,384,512]:
    Pk = Vh[:k].T@Vh[:k]; rep(f'Wv restricted to top-{k} feat PCs', Wv0@Pk, Wt0)
