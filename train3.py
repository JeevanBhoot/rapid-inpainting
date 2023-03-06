import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import BatchSampler, RandomSampler
import torch.nn as nn
import wandb
import os
from datetime import datetime
import math
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, MobileViTForSemanticSegmentation, MobileViTFeatureExtractor

from model import *
from loss import *
from dataset import *
from positionalembedding import *

np.random.seed(0)
torch.manual_seed(0)
wandb.init(project='inpaint', name='mobilevit-x1')
filepath = '/data/cornucopia/jsb212/seg-dataset/all_trya_imgs'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32
num_epochs = 100
lr = 0.001
pos_embedding = True
normalise = False
height = width = 256

now = datetime.now()
now_str = './outputs/' + now.strftime('%d-%m-%Y_%H-%M')
os.makedirs(now_str, exist_ok=True)
os.makedirs(f'{now_str}/imgs_epochs_train', exist_ok=True)
os.makedirs(f'{now_str}/imgs_epochs_val', exist_ok=True)

# Define the loss function
loss_fn = WeightedLoss([VGGLoss(),
                        nn.MSELoss(),
                        TVLoss(p=1)],
                        [1, 30, 10]).to(device) #1, 40, 10
#mse = nn.MSELoss()

#Transforms
# transform_list = [
#     transforms.Resize((512, 512)),
#     transforms.RandomResizedCrop((512, 512)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
#     transforms.ToTensor()
# ]
img_transforms_lst = [
    transforms.Resize((height, width)),
    transforms.ToTensor()
]
if normalise:
    img_transforms_lst.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    invNormalise = InvNormalise(device=device)
img_transform = transforms.Compose(img_transforms_lst)

feature_extractor = MobileViTFeatureExtractor.from_pretrained("apple/deeplabv3-mobilevit-xx-small")

#Load data
data = ImgMaskDataset(filepath, img_transform)
num_samples = len(data.imgs)
num_train = int(0.8 * num_samples)
num_val = num_samples - num_train
train_data, val_data = random_split(data, [num_train, num_val])
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=True)

#Initialise model
model = MobileViTForSemanticSegmentation.from_pretrained("apple/deeplabv3-mobilevit-xx-small")
model.segmentation_head.classifier = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0),
                                                   nn.LeakyReLU(),
                                                   nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0),
                                                   nn.LeakyReLU(),
                                                   nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0),
                                                   nn.LeakyReLU(),
                                                   nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, output_padding=0),
                                                   nn.Tanh())


# model.decode_head.classifier = InpaintingHead2()
if pos_embedding:
    dim = 64
    pe = positionalencoding2d(dim, height, width)
    pe = pe.to(device)
    model.mobilevit.conv_stem.convolution = nn.Conv2d(in_channels=3+dim, out_channels=16, 
                                                                kernel_size=3, stride=2, padding=1)
    
if torch.cuda.device_count() > 1:
  print(torch.cuda.device_count(), "GPUs")
  model = nn.DataParallel(model)

model.to(device)
model_params = model.state_dict()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)

num_iters = math.ceil(num_samples/batch_size) * num_epochs

# Train the model
loss_val = 10000
with tqdm(total=num_iters) as pbar:
    for epoch in range(num_epochs):
        # Training loop
        model.train()
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            if normalise:
                imagesT = invNormalise(images)
            masksX = masks.unsqueeze(1).repeat(1, 3, 1, 1)
            images_masked = images * masksX
            if pos_embedding:
                peX = pe.repeat(images_masked.shape[0], 1, 1, 1)
                images_masked = torch.cat((images_masked, peX), 1)
            images_masked = images_masked.to(device)
            optimizer.zero_grad()
            outputs = model(images_masked).logits
            if normalise:
                outputs = invNormalise(outputs)
                images = imagesT
            loss = loss_fn(outputs, images)
            loss.backward()
            optimizer.step()
            pbar.update(1)
            torch.cuda.empty_cache()
        
        if epoch % 10 == 0:
            save_array_as_img(array=outputs[0], filepath=f'{now_str}/imgs_epochs_train/epoch_{epoch}.png')

        # Validation loop
        model.eval()
        with torch.no_grad():
           for i, (images, masks) in enumerate(val_loader):
                images, masks = images.to(device), masks.to(device)
                if normalise:
                    imagesT = invNormalise(images)
                masksX = masks.unsqueeze(1).repeat(1, 3, 1, 1)
                images_masked = images * masksX
                if pos_embedding:
                    peX = pe.repeat(images_masked.shape[0], 1, 1, 1)
                    images_masked = torch.cat((images_masked, peX), 1)
                images_masked = images_masked.to(device)
                outputs = model(images_masked).logits
                if normalise:
                    outputs = invNormalise(outputs)
                    images = imagesT
                val_loss = loss_fn(outputs, images)
                pbar.update(1)
                torch.cuda.empty_cache()

        if epoch % 10 == 0:
            save_array_as_img(array=outputs[0], filepath=f'{now_str}/imgs_epochs_val/epoch_{epoch}.png')

        current_lr = scheduler.get_last_lr()[0]
        print("Epoch {}: Train Loss: {}, Val Loss: {}".format(epoch+1, loss.item(), val_loss.item()))
        wandb.log({'Train Loss': loss.item(), 'Val Loss': val_loss.item(), 'Learning Rate': current_lr})

        if loss.item() < loss_val:
            model_params = model.state_dict()

        scheduler.step()

for i, output in enumerate(outputs):
    save_array_as_img(array=output, filepath=f'{now_str}/output_{i}.png')

    input = images[i]
    save_array_as_img(array=input, filepath=f'{now_str}/input_{i}.png')

    mask = masks[i]
    save_array_as_img(array=mask, filepath=f'{now_str}/mask_{i}.png')

    final_output = input * mask + output * (1-mask)
    save_array_as_img(array=final_output, filepath=f'{now_str}/final_img_{i}.png')

torch.save(model_params, f'{now_str}/mobilevit.pth')

