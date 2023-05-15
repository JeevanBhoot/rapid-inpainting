import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import os
from transformers import MobileViTForSemanticSegmentation, MobileViTFeatureExtractor

from model import *
from loss import *
from dataset import *
from positionalembedding import *

np.random.seed(0)
torch.manual_seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
filepath = '/data/cornucopia/jsb212/seg-dataset/tryon-test-pics'
modelpath = '/home/mifs/jsb212/encoder-inpaint/outputs/2023-03-15_16-52'

pos_embedding = 64
normalise = False
height = width = 256

model = MobileViTForSemanticSegmentation.from_pretrained("apple/deeplabv3-mobilevit-xx-small")
model.segmentation_head.classifier = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0),
                                                   nn.LeakyReLU(),
                                                   nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0),
                                                   nn.LeakyReLU(),
                                                   nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0),
                                                   nn.LeakyReLU(),
                                                   nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, output_padding=0),
                                                   nn.Tanh())

if pos_embedding:
    pe = positionalencoding2d(pos_embedding, height, width)
    pe = pe.to(device)
    model.mobilevit.conv_stem.convolution = nn.Conv2d(in_channels=3+pos_embedding, out_channels=16, 
                                                                kernel_size=3, stride=2, padding=1)
    
model.load_state_dict(torch.load(f'{modelpath}/mobilevit.pth'))
model.to(device)
model.eval()

img_transforms_lst = [
        transforms.Resize((height, width)),
        transforms.ToTensor()
    ]
img_transform = transforms.Compose(img_transforms_lst)

modelpath += '/inference'
os.makedirs(f'{modelpath}', exist_ok=True)

for i in range(10):
    print(i)
    img = Image.open(f'{filepath}/images/test{i}.png')
    img_t = img_transform(img)

    mask = Image.open(f'{filepath}/masks/mask{i}.png')
    mask_t = img_transform(mask)

    img_t, mask_t = img_t.to(device).unsqueeze(0), mask_t.to(device).unsqueeze(0)
    img_t_masked = img_t * mask_t
    if pos_embedding:
        pe = pe.repeat(img_t_masked.shape[0], 1, 1, 1)
        img_t_masked = torch.cat((img_t_masked, pe), 1)
    img_t_masked = img_t_masked.to(device)

    output = model(img_t_masked).logits
    output, mask_t, img_t = output[0], mask_t[0], img_t[0]

    #direct replacement of non-masked pixels
    final_output = img_t * mask_t + output * (1-mask_t)

    save_array_as_img(array=output, filepath=f'{modelpath}/output_{i}.png')
    save_array_as_img(array=img_t, filepath=f'{modelpath}/input_{i}.png')
    #save_array_as_img(array=mask_t, filepath=f'{modelpath}/mask.png')
    save_array_as_img(array=final_output, filepath=f'{modelpath}/finaloutput_{i}.png')