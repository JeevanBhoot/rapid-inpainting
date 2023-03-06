import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from model import *
from loss import *
from dataset import *
from PIL import Image
import time
import numpy as np

device = 'cuda' if torch.cuda.is_available else 'cpu'
if device == 'cuda':
    torch.cuda.empty_cache()

model = LightweightEncoderDecoder()
model.load_state_dict(torch.load('model6.pth'))
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

mask = generate_random_mask(512, 512)
mask_t = torch.tensor(mask)

img_t, mask_t = img_t.to(device).unsqueeze(0), mask_t.to(device).unsqueeze(0)

n=500
start_time = time.time()
for _ in range(n):
    output = model(img_t, mask_t)
total_time = time.time() - start_time
print('Total Time: ', total_time)
print('Time per Image: ', total_time/n)
print('Frame Rate: ', n/total_time)

output, mask_t, img_t = output[0], mask_t[0], img_t[0]

arr = (output.cpu().detach().numpy() * 255).astype(np.uint8)
output_img = Image.fromarray(arr.T)
output_img.save('output.png')

arr_img = (img_t.cpu().detach().numpy() * 255).astype(np.uint8)
input_img = Image.fromarray(arr_img.T)
input_img.save('input_img.png')

arr_mask = (mask_t.cpu().detach().numpy() * 255).astype(np.uint8)
mask_img = Image.fromarray(arr_mask.T)
mask_img.save('mask.png')

#direct replacement of non-masked pixels
#mask_t = mask_t.expand_as(img_t)
print(output.shape)
final_output = img_t * mask_t + output * (1-mask_t)
final_arr = (final_output.cpu().detach().numpy() * 255).astype(np.uint8)
final_img = Image.fromarray(final_arr.T)
final_img.save('final_img.png')