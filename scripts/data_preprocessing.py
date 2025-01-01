import numpy as np
import pandas as pd
import json
import os
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage
from torchvision import transforms
from tqdm import tqdm
import argparse as ap

#Constants and Hyperparameters
mean = torch.tensor([0.485, 0.456, 0.406])
std_dev = torch.tensor([0.229, 0.224, 0.225])
image_size = 224
batches_size = 512
number_of_workers = 4
Device = 'cuda'
destination_dir="/kaggle/working/preprocessed_images"
output_dir = "processed_images"
os.makedirs(output_dir, exist_ok=True) 

to_pil = ToPILImage()

#Directory paths used
captions_link = ''  # /coco2017/annotations/captions_train2017.json'
train_inp_path = '' # /coco2017/train2017

class Initialize():
    def initialize(self):
        with open(captions_link, 'r') as file:
            data = json.load(file)
        data_df = pd.json_normalize(data, "annotations")
        data_df['image_id'] = data_df['image_id'].apply(lambda x : os.path.join(train_inp_path, ("{:012}".format(x) + ".jpg")))
        img_df = pd.DataFrame(data_df['image_id'].unique(), columns=['image_path'])
        return img_df

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image at {img_path}: {e}")
        if self.transform:
            image = self.transform(image)
        return image, img_path

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__=="__main__":
    initial = Initialize()

    parser1, parser2 = ap.ArgumentParser()
    parser1.add_argument("--saving", help="1 if to save untransformed images to disk, esle 0")
    args1 = parser1.parse_args()
    parser2.add_argument("--transformed_save", help="1 if to save transformed images to disk, esle 0")
    args2 = parser2.parse_args()

    img_df = initial.initialize()
    images_list = np.squeeze(img_df.values.tolist())
    dataset = CustomImageDataset(image_paths=images_list, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batches_size, shuffle=True, num_workers=number_of_workers, pin_memory=True)

    device = Device
    total_batches = len(dataloader)
    saving = 1
    transformed_save = 1
    for batch_idx, (images, img_paths) in enumerate(tqdm(dataloader, total=total_batches, desc="Processing Batches")):
        images_gpu = images.to("cuda")
        if(args1.saving == 1):
            images_cpu = images.cpu()
            if(transformed_save == 1):
                images_cpu = images_cpu * std_dev[:, None, None] + mean[:, None, None]
            for img_idx, image_tensor in enumerate(images_cpu):
                pil_image = to_pil(image_tensor)
                original_filename = os.path.basename(img_paths[img_idx])
                save_path = os.path.join(output_dir, f"{batch_idx}_{img_idx}_{original_filename}")
                pil_image.save(save_path)
    if(args2.transformed_save == 1):
        print("All images processed to tensors and transformed images saved to disk successfully")
    else:
        print("All images processed to tensors and untransformed images saved to disk successfully")