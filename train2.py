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
wandb.init(project='inpaint', name='basic-2')
filepath = '/data/cornucopia/jsb212/seg-dataset/test2'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 4
num_epochs = 1000
lr = 0.001
pos_embedding = True
alpha = 0
beta = 0
load_pretrained = False
normalise = False
height = width = 256

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
styleloss = StyleLoss(device=device)
criterionD = nn.BCELoss()
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

#Load data
data = ImgMaskDataset(filepath, img_transform)
# sampler = BatchSampler(
#     RandomSampler(data),
#     batch_size=batch_size,
#     drop_last=False)
num_samples = len(data.imgs)
#num_train = int(0.8 * num_samples)
#num_val = num_samples - num_train
#train_data, val_data = random_split(data, [num_train, num_val])
train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
#val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

#Initialise model
model = smp.Unet(
    encoder_name="mobilenet_v2",
    encoder_weights="imagenet",
    classes=3,
)
model.segmentation_head = InpaintingHead()
if pos_embedding:
    dim = 64
    pe = positionalencoding2d(dim, height, width)
    pe = pe.to(device)
    model.encoder.features[0][0] = nn.Conv2d(in_channels=3+dim, out_channels=32, 
                                         kernel_size=3, stride=2, padding=1)
model.to(device)

#xmodel = XModel(model=model)
#xmodel.to(device)

model_params = model.state_dict()

#modelD = Discriminator()
#initialise_model(modelD, device)

#if load_pretrained:
#    model.load_state_dict(torch.load('outputs/16-02-2023_18-10/generator.pth'))
#    modelD.load_state_dict(torch.load('outputs/16-02-2023_18-10/discriminator.pth'))

#num_params = sum(p.numel() for p in model.parameters())
#print(num_params)


# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#optimizerD = torch.optim.Adam(modelD.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)

num_iters = math.ceil(num_samples/batch_size) * num_epochs

# Train the model
loss_val = 10000
with tqdm(total=num_iters) as pbar:
    for epoch in range(num_epochs):
        # Training loop
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            if normalise:
                imagesT = invNormalise(images)
            masksX = masks.unsqueeze(1).repeat(1, 3, 1, 1)
            images_masked = images * masksX
            if pos_embedding:
                peX = pe.repeat(images_masked.shape[0], 1, 1, 1)
                images_masked = torch.cat((images_masked, peX), 1).to(device)

            #optimizerD.zero_grad()
            # TRAIN DISCRIMINATOR: maximize log(D(x)) + log(1 - D(G(z)))
            # (1) Train with all real batch: log(D(x))
            #labels_real = torch.full((images.shape[0],), 1, dtype=torch.float, device=device) #label=1 true
            #preds_real = modelD(images).view(-1)
            #lossD = criterionD(preds_real, labels_real)

            # (2) Train with all fake batch: log(1 - D(G(z)))
            #outputs = model(images_masked)
            #labels_fake = torch.full((images.shape[0],), 0, dtype=torch.float, device=device) #label=0 fake
            #preds_fake = modelD(outputs).view(-1)
            #lossD += criterionD(preds_fake, labels_fake)
            #lossD.backward()
            #optimizerD.step()
            
            #TRAIN GENERATOR: maximize log(D(G(z)))
            optimizer.zero_grad()
            outputs = model(images_masked)
            if normalise:
                outputsT = invNormalise(outputs)
            #preds_fake = modelD(outputs).view(-1)
            #for i, output in enumerate(outputs):
            #    image = images[i]
            #    mask = masksX[i]
            #    output_maskselect, images_maskselect =  torch.masked_select(output, mask.eq(0)), torch.masked_select(image, mask.eq(0))
            #    num_elements = output_maskselect.shape[0]
            #outputs_maskselect, images_maskselect = torch.masked_select(outputs, masks.eq(0)), torch.masked_select(images, masks.eq(0))
            #num_elements = output_maskselect.shape[0]
            #output_maskselect, images_maskselect = output_maskselect.view(-1, 3, 512, 512)[:, :, :num_elements], images_maskselect.view(-1, 3, 512, 512)[:, :, :num_elements]
            #print(outputs.size(), images.size())
            #print(output_maskselect.size(), images_maskselect.size())
            if normalise:
                outputs, images = outputsT, imagesT
            output_comp = images * masksX + outputs * (1-masksX) #composite output - combine nonmasked area of gt with masked area of generation
            loss = criterion(outputs, images)
            #loss += beta*(styleloss(outputs, images) + styleloss(output_comp, images))
            #loss += alpha*criterionD(preds_fake, labels_real)
            loss.backward()
            optimizer.step()
            pbar.update(1)
            torch.cuda.empty_cache()

        if epoch % 10 == 0:
            save_array_as_img(array=outputs[0], filepath=f'{now_str}/imgs_epochs/epoch_{epoch}.png')
        current_lr = scheduler.get_last_lr()[0]
        #print("Epoch {}: Generator Train Loss: {}   Discriminator Train Loss: {}".format(epoch+1, loss.item(), lossD))
        #wandb.log({'Generator Train Loss': loss.item(), 'Discriminator Train Loss': lossD, 'Learning Rate': current_lr})
        print("Epoch {}: Generator Train Loss: {}".format(epoch+1, loss.item()))
        wandb.log({'Generator Train Loss': loss.item(), 'Learning Rate': current_lr})
        if loss.item() < loss_val:
            model_params = model.state_dict()
            loss_val = loss.item()
        scheduler.step()
        # Validation loop
        #with torch.no_grad():
        #    for i, (images, masks) in enumerate(val_loader):
        #        output = model(images.cuda(), masks.cuda())
        #        val_loss = criterion(output, images)
        #print("Epoch {}: Train Loss: {}, Val Loss: {}".format(epoch+1, loss.item(), val_loss.item()))
        #wandb.log({'Train Loss': loss.item(), 'Val Loss': val_loss.item()})


for i, output in enumerate(outputs):
    save_array_as_img(array=output, filepath=f'{now_str}/output_{i}.png')

    input = images[i]
    save_array_as_img(array=input, filepath=f'{now_str}/input_{i}.png')

    mask = masks[i]
    save_array_as_img(array=mask, filepath=f'{now_str}/mask_{i}.png')

    final_output = input * mask + output * (1-mask)
    save_array_as_img(array=final_output, filepath=f'{now_str}/final_img_{i}.png')

torch.save(model_params, f'{now_str}/generator.pth')
#torch.save(modelD.state_dict(), f'{now_str}/discriminator.pth')

