import torch, typo
from torch.utils.data import DataLoader
for name, lo, hi, seed0 in [('TA', 10, 14, 100000), ('TB', 14, 18, 200000)]:
    C, A, Y = [], [], []
    for c, a, y in DataLoader(typo.DS(typo.train_split(lo, hi), seed0), batch_size=250, num_workers=16): C.append(c.half()); A.append(a.half()); Y.append(y)
    torch.save((torch.cat(C), torch.cat(A), torch.cat(Y)), f'typo/{name}.pt'); print(name, torch.cat(Y).shape, flush=True)
