import torch
import torch.utils.data
from torchvision import transforms
from PIL import Image
from mask import generate_random_mask
from pathlib import Path
import numpy as np

def get_image_files(filepath):
    extensions = ['png', 'jpg', 'jpeg']
    result = []
    for extension in extensions:
        result += list(Path(filepath).glob(f'**/*.{extension}'))
    return result

def save_array_as_img(array, filepath):
    arr = (array.cpu().detach().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr.T)
    img.save(filepath)

class ImgMaskDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, img_transform):
        super(ImgMaskDataset, self).__init__()

        self.img_transform = img_transform
        self.imgs = get_image_files(dataset_path)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = Image.open(img_path)
        img_t = self.img_transform(img)

        mask = generate_random_mask(img_t.size()[1], img_t.size()[2])
        mask_t = torch.tensor(mask)

        return img_t, mask_t

    def __len__(self):
        return len(self.imgs)

class ImgMaskDataset2(torch.utils.data.Dataset):
    def __init__(self, dataset_path, img_transform):
        super(ImgMaskDataset2, self).__init__()

        self.img_transform = img_transform
        self.imgs = get_image_files(dataset_path)

    def __getitem__(self, index):
        img_paths = [self.imgs[i] for i in index]
        imgs = torch.Tensor()
        for img_path in img_paths:
            img = Image.open(img_path)
            img_t = self.img_transform(img).unsqueeze(0)
            imgs = torch.cat((imgs, img_t))
        mask = generate_random_mask(512, 512)
        mask_t = torch.tensor(mask)
        mask_t = mask_t.unsqueeze(0).repeat(3,1,1).unsqueeze(0).repeat(len(index),1,1,1)

        return imgs, mask_t

    def __len__(self):
        return len(self.imgs)

class InvNormalise(torch.nn.Module):
    "Pass a batch of tensors."
    def __init__(self, device):
        super(InvNormalise, self).__init__()
        self.device = device
        self.transform = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                   transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])])
    def forward(self, batch):
        inv_tensors = []
        for tensor in batch:
            inv_tensor = self.transform(tensor)
            inv_tensors.append(inv_tensor.to(self.device))
        return torch.stack(inv_tensors).to(self.device)
