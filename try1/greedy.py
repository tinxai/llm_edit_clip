from wsp import *
import json
S = load('search'); f,y = S
ia = torch.tensor([i for i in range(10000) if i%10<5],device=dev); ib = torch.tensor([i for i in range(10000) if i%10>=5],device=dev)
A=(f[ia],y[ia]); B=(f[ib],y[ib]); V=load('val')
Wv,Wt = Wv0.clone(), Wt0.clone(); a0,b0 = acc(Wv,Wt,A),acc(Wv,Wt,B)
print('start A %.2f B %.2f | val %.2f'%(a0,b0,acc(Wv,Wt,V)),flush=True)
log=[]
scales=[0.0,0.5,0.8,1.25,1.5]
for rnd in range(15):
    ca,cb = acc(Wv,Wt,A),acc(Wv,Wt,B); best=None
    for side,W,n in [('v',Wv,768),('t',Wt,512)]:
        for d in range(n):
            for s in scales:
                W2=W.clone(); W2[:,d]*=s
                a,b = (acc(W2,Wt,A),acc(W2,Wt,B)) if side=='v' else (acc(Wv,W2,A),acc(Wv,W2,B))
                if a>ca and b>=cb and (best is None or (a-ca)+(b-cb)>best[0]): best=((a-ca)+(b-cb),side,d,s)
    if best is None: print('no accepted edit in round',rnd); break
    _,side,d,s = best
    (Wv if side=='v' else Wt)[:,d]*=s
    r=dict(round=rnd,edit=f'W{side}[:,{d}]*={s}',A=acc(Wv,Wt,A),B=acc(Wv,Wt,B),val=acc(Wv,Wt,V)); log.append(r); print(r,flush=True)
torch.save({'Wv':Wv.cpu(),'Wt':Wt.cpu()},'edited.pt'); json.dump(log,open('greedy_log.json','w'),indent=1)
