import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from model import *
from loss import *
from dataset import *
from positionalembedding import *
from PIL import Image
import time
import numpy as np
import segmentation_models_pytorch as smp
import os

seed = 1
pos_embedding = True

device = 'cuda' if torch.cuda.is_available else 'cpu'
if device == 'cuda':
    torch.cuda.empty_cache()

model = smp.Unet(
    encoder_name="mobilenet_v2",
    encoder_weights="imagenet",
    classes=3,
)
model.segmentation_head = InpaintingHead()
if pos_embedding:
    dim = 64
    pe = positionalencoding2d(dim, 512, 512)
    #pe = pe.repeat(batch_size, 1, 1, 1)
    pe = pe.unsqueeze(0)
    pe = pe.to(device)
    model.encoder.features[0][0] = nn.Conv2d(in_channels=3+dim, out_channels=32, 
                                         kernel_size=3, stride=2, padding=1)
model.load_state_dict(torch.load('unet-mobilenetv2-pe-4.pth'))
model.eval()
model.to(device)

filepath = '/data/cornucopia/jsb212/seg-dataset/test1'

img = Image.open(filepath + '/0.jpg')
img_transforms_lst = [
    transforms.Resize((512, 512)),
    transforms.ToTensor()
]
img_transform = transforms.Compose(img_transforms_lst)
img_t = img_transform(img)

np.random.seed(seed=seed)
mask = generate_random_mask(512, 512)
mask_t = torch.tensor(mask)

img_t, mask_t = img_t.to(device).unsqueeze(0), mask_t.to(device).unsqueeze(0)
img_t_masked = img_t * mask_t
if pos_embedding:
    img_t_masked = torch.cat((img_t_masked, pe), 1)
img_t_masked = img_t_masked.to(device)

n=500
start_time = time.time()
for _ in range(n):
    output = model(img_t_masked)
total_time = time.time() - start_time
print('Total Time: ', total_time)
print('Time per Image: ', total_time/n)
print('Frame Rate: ', n/total_time)

output, mask_t, img_t = output[0], mask_t[0], img_t[0]

os.makedirs('outputs', exist_ok=True)

arr = (output.cpu().detach().numpy() * 255).astype(np.uint8)
output_img = Image.fromarray(arr.T)
output_img.save('./outputs/output-unet-pe4.png')

#arr_img = (img_t.cpu().detach().numpy() * 255).astype(np.uint8)
#input_img = Image.fromarray(arr_img.T)
#input_img.save('input_img.png')

arr_mask = (mask_t.cpu().detach().numpy() * 255).astype(np.uint8)
mask_img = Image.fromarray(arr_mask.T)
mask_img.save('./outputs/mask-unet-pe4.png')

#direct replacement of non-masked pixels
#mask_t = mask_t.expand_as(img_t)
final_output = img_t * mask_t + output * (1-mask_t)
final_arr = (final_output.cpu().detach().numpy() * 255).astype(np.uint8)
final_img = Image.fromarray(final_arr.T)
final_img.save('./outputs/final_img-unet-pe4.png')