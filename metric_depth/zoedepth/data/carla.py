import os
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from PIL import Image
from torchvision import transforms as T

class CarlaDataset(Dataset):
    def __init__(self, data_path) -> None:
        path = data_path['carla_root']

        print("PATH:", path)

        self.image_path = os.path.join(path, 'img')
        self.depth_path = os.path.join(path, 'depth')

        self.image_files = sorted(os.listdir(self.image_path))
        self.depth_files = sorted(os.listdir(self.depth_path))

        self.resize = T.Resize(100)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index) -> Any:
        png = Image.open(os.path.join(self.image_path, self.image_files[index])).convert('RGBA')
        background = Image.new('RGBA', png.size, (255,255,255))
        image = alpha_composite = Image.alpha_composite(background, png).convert('RGB')
        # read_image(os.path.join(self.image_path, self.image_files[index]))
        image = T.ToTensor()(image)
        depth = read_image(os.path.join(self.depth_path, self.depth_files[index]))

        image, depth = self.resize(image.to(torch.float)), self.resize(depth.to(float))

        return dict(image=image, depth=depth, dataset=['carla'])

def get_carla_loader(data_dir_root, batch_size=1, **kwargs):
    dataset = CarlaDataset(data_path=data_dir_root)
    return DataLoader(dataset, batch_size, **kwargs)

def main():
    dataset = CarlaDataset({'carla_root': "/data/carla"})
    batch = dataset.__getitem__(0)
    a=1

if __name__ == '__main__':
    main()