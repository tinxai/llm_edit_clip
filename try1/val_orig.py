import torch, torch.nn.functional as F, harness as H
from wsp import classifier, Wt0
m = H.load_model(); Wc = classifier(Wt0); z, y = [], []
with torch.no_grad():
    for x, yy in H.loader(H.val_items()):
        with torch.autocast('cuda', dtype=torch.float16): z.append(F.normalize(m.visual_projection(m.vision_model(pixel_values=x.to('cuda')).pooler_output).float(), dim=-1))
        y.append(yy)
z = torch.cat(z); y = torch.cat(y).cuda(); print('VAL original fp16 %.2f' % (((z @ Wc.T).argmax(1) == y).float().mean().item()*100))
