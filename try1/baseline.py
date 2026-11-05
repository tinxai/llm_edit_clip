import torch, time, harness as H
m = H.load_model(); W = H.text_classifier(m)
for name, items in [('search(train 10/cls)', H.search_items()), ('val', H.val_items())]:
    t = time.time(); f, y = H.encode_images(m, H.loader(items))
    print(name, len(items), 'top1 = %.2f' % H.top1(f, y, W), '%.0fs' % (time.time() - t), flush=True)
