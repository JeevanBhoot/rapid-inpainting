import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from model import *
from loss import *
from dataset import *
import wandb

wandb.init(project='inpaint', name='4')
filepath = '/data/cornucopia/jsb212/seg-dataset/test1'
device = 'cuda' if torch.cuda.is_available else 'cpu'
batch_size = 8
num_epochs = 10000
lr = 0.001

# Define the loss function
loss_fn = WeightedLoss([VGGLoss(),
                        nn.MSELoss(),
                        TVLoss(p=1)],
                        [1, 40, 10]).to(device) #1, 40, 10

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
    transforms.Resize((512, 512)),
    transforms.ToTensor()
]
img_transform = transforms.Compose(img_transforms_lst)

#Load data
data = ImgMaskDataset(filepath, img_transform)
num_samples = len(data.imgs)
#num_train = int(0.8 * num_samples)
#num_val = num_samples - num_train
#train_data, val_data = random_split(data, [num_train, num_val])
train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
#val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

#Initialise model
model = LightweightEncoderDecoder()
model.to(device)
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9997)

# Train the model
for epoch in range(num_epochs):
    # Training loop
    for i, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        output = model(images, masks)
        loss = loss_fn(output, images)
        loss.backward()
        optimizer.step()
    current_lr = scheduler.get_last_lr()[0]
    print("Epoch {}: Train Loss: {}".format(epoch+1, loss.item()))
    wandb.log({'Train Loss': loss.item(), 'Learning Rate': current_lr})
    scheduler.step()
    # Validation loop
    #with torch.no_grad():
    #    for i, (images, masks) in enumerate(val_loader):
    #        output = model(images.cuda(), masks.cuda())
    #        val_loss = loss_fn(output, images)
    #print("Epoch {}: Train Loss: {}, Val Loss: {}".format(epoch+1, loss.item(), val_loss.item()))
    #wandb.log({'Train Loss': loss.item(), 'Val Loss': val_loss.item()})

torch.save(model.state_dict(), 'model7.pth')

