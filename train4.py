import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import BatchSampler, RandomSampler
import torch.nn as nn
import wandb
import segmentation_models_pytorch as smp
import os
from datetime import datetime
import math
from tqdm import tqdm

from model import *
from loss import *
from dataset import *
from positionalembedding import *

np.random.seed(0)
torch.manual_seed(0)
wandb.init(project='inpaint', name='unet-discrim-7')
filepath = '/data/cornucopia/jsb212/seg-dataset/test2'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 4
num_epochs = 400
lr = 0.001
pos_embedding = True
alpha = 0.5
load_pretrained = False

now = datetime.now()
now_str = './outputs/' + now.strftime('%d-%m-%Y_%H-%M')
os.makedirs(now_str, exist_ok=True)
os.makedirs(f'{now_str}/imgs_epochs', exist_ok=True)

# Define the loss function
criterion = WeightedLoss([VGGLoss(),
                        nn.MSELoss(),
                        TVLoss(p=1)],
                        [1, 30, 10]).to(device) #1, 40, 10
mse = nn.MSELoss()
criterionD = nn.BCELoss()

img_transforms_lst = [
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
#transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
img_transform = transforms.Compose(img_transforms_lst)
invNormalise = InvNormalise(device=device)

#Load data
data = ImgMaskDataset(filepath, img_transform)
num_samples = len(data.imgs)
train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

#Initialise model
model = smp.Unet(
    encoder_name="mobilenet_v2",
    encoder_weights="imagenet",
    classes=3,
)
model.segmentation_head = InpaintingHead()
if pos_embedding:
    dim = 64
    pe = positionalencoding2d(dim, 512, 512)
    pe = pe.to(device)
    model.encoder.features[0][0] = nn.Conv2d(in_channels=3+dim, out_channels=32, 
                                         kernel_size=3, stride=2, padding=1)
model.to(device)

modelD = Discriminator()
initialise_model(modelD, device)

if load_pretrained:
    model.load_state_dict(torch.load('outputs/16-02-2023_18-10/generator.pth'))
    modelD.load_state_dict(torch.load('outputs/16-02-2023_18-10/discriminator.pth'))

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizerD = torch.optim.Adam(modelD.parameters(), lr=lr)

num_iters = math.ceil(num_samples/batch_size) * num_epochs * 2

# Train the model
with tqdm(total=num_iters) as pbar:
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    for epoch in range(num_epochs):
        # Training loop
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            imagesT = invNormalise(images.float())
            masksX = masks.unsqueeze(1).repeat(1, 3, 1, 1)
            images_masked = images * masksX
            if pos_embedding:
                peX = pe.repeat(images_masked.shape[0], 1, 1, 1)
                images_masked = torch.cat((images_masked, peX), 1)
            images_masked = images_masked.to(device)

            optimizerD.zero_grad()
            # TRAIN DISCRIMINATOR: maximize log(D(x)) + log(1 - D(G(z)))
            # (1) Train with all real batch: log(D(x))
            labels_real = torch.full((images.shape[0],), 1, dtype=torch.float, device=device) #label=1 true
            preds_real = modelD(images).view(-1)
            lossD = criterionD(preds_real, labels_real)

            # (2) Train with all fake batch: log(1 - D(G(z)))
            outputs = model(images_masked)
            labels_fake = torch.full((images.shape[0],), 0, dtype=torch.float, device=device) #label=0 fake
            preds_fake = modelD(outputs).view(-1)
            lossD += criterionD(preds_fake, labels_fake)
            lossD.backward()
            optimizerD.step()
            
            #TRAIN GENERATOR WITHOUT DISCRIMINATOR LOSS
            optimizer.zero_grad()
            outputs = model(images_masked)
            outputsT = invNormalise(outputs)
            preds_fake = modelD(outputs).view(-1)
            loss = criterion(outputsT, imagesT)
            loss.backward()
            optimizer.step()
            pbar.update(1)
            torch.cuda.empty_cache()

        if epoch % 10 == 0:
            save_array_as_img(array=outputsT[0], filepath=f'{now_str}/imgs_epochs/stage1_epoch_{epoch}.png')
        current_lr = scheduler.get_last_lr()[0]
        print("Stage I, Epoch {}: Generator Train Loss: {}   Discriminator Train Loss: {}".format(epoch+1, loss.item(), lossD))
        wandb.log({'Generator Train Loss': loss.item(), 'Discriminator Train Loss': lossD, 'Learning Rate': current_lr})
        scheduler.step()

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    for epoch in range(num_epochs):
        # Training loop
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            imagesT = invNormalise(images.float())
            masksX = masks.unsqueeze(1).repeat(1, 3, 1, 1)
            images_masked = images * masksX
            if pos_embedding:
                peX = pe.repeat(images_masked.shape[0], 1, 1, 1)
                images_masked = torch.cat((images_masked, peX), 1)
            images_masked = images_masked.to(device)

            optimizerD.zero_grad()
            # TRAIN DISCRIMINATOR: maximize log(D(x)) + log(1 - D(G(z)))
            # (1) Train with all real batch: log(D(x))
            labels_real = torch.full((images.shape[0],), 1, dtype=torch.float, device=device) #label=1 true
            preds_real = modelD(images).view(-1)
            lossD = criterionD(preds_real, labels_real)

            # (2) Train with all fake batch: log(1 - D(G(z)))
            outputs = model(images_masked)
            labels_fake = torch.full((images.shape[0],), 0, dtype=torch.float, device=device) #label=0 fake
            preds_fake = modelD(outputs).view(-1)
            lossD += criterionD(preds_fake, labels_fake)
            lossD.backward()
            optimizerD.step()
            
            #TRAIN GENERATOR with discrim loss: maximize log(D(G(z)))
            optimizer.zero_grad()
            outputs = model(images_masked)
            outputsT = invNormalise(outputs)
            preds_fake = modelD(outputs).view(-1)
            loss = criterion(outputsT, imagesT)
            loss += alpha*criterionD(preds_fake, labels_real)
            loss.backward()
            optimizer.step()
            pbar.update(1)
            torch.cuda.empty_cache()

        if epoch % 10 == 0:
            save_array_as_img(array=outputsT[0], filepath=f'{now_str}/imgs_epochs/stage2_epoch_{epoch}.png')
        current_lr = scheduler.get_last_lr()[0]
        print("Stage II, Epoch {}: Generator Train Loss: {}   Discriminator Train Loss: {}".format(epoch+1, loss.item(), lossD))
        wandb.log({'Generator Train Loss': loss.item(), 'Discriminator Train Loss': lossD, 'Learning Rate': current_lr})
        scheduler.step()


for i, output in enumerate(outputsT):
    save_array_as_img(array=output, filepath=f'{now_str}/output_{i}.png')

    input = imagesT[i]
    save_array_as_img(array=input, filepath=f'{now_str}/input_{i}.png')

    mask = masks[i]
    save_array_as_img(array=mask, filepath=f'{now_str}/mask_{i}.png')

    final_output = input * mask + output * (1-mask)
    save_array_as_img(array=final_output, filepath=f'{now_str}/final_img_{i}.png')

torch.save(model.state_dict(), f'{now_str}/generator.pth')
torch.save(modelD.state_dict(), f'{now_str}/discriminator.pth')

