"""The winning edit: switch off two attention heads of CLIP ViT-B/32's vision tower (no training).
head h of layer l = columns [h*64:(h+1)*64] of that layer's out_proj weight."""
import torch
def apply(clip_model):
    layers = clip_model.vision_model.encoder.layers
    with torch.no_grad():
        layers[11].self_attn.out_proj.weight[:, 512:576] = 0   # layer 11, head 8
        layers[8].self_attn.out_proj.weight[:, 256:320] = 0    # layer 8,  head 4
    return clip_model
if __name__ == '__main__':
    from transformers import CLIPModel
    m = apply(CLIPModel.from_pretrained('openai/clip-vit-base-patch32')); m.save_pretrained('clip_b32_zero2heads')
