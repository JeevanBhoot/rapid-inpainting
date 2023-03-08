from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from model import *
from loss import *
from dataset import *

# img_transforms_lst = [
#     transforms.Resize((512, 512)),
#     transforms.ToTensor()
# ]
# mask_transforms_lst = [
#     transforms.ToTensor()
# ]
# img_transform = transforms.Compose(img_transforms_lst)
# mask_transform = transforms.Compose(mask_transforms_lst)

# img = Image.load('0.png')
# img_t = img_transform(img)
# mask = generate_random_mask(512, 512)
# mask_t = mask_transform(mask)

# x = img_t * mask_t

print(get_image_files('/data/cornucopia/jsb212/seg-dataset/eval_inpaint'))