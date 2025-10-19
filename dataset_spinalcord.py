import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random

class SpinalCordDataset(Dataset):
    def __init__(self, base_dir, split, transform=None, low_res_size=(64, 64)):
        """
        Args:
            base_dir (string): Directory with all the data (e.g., 'C:/.../segmentation dataset/').
            split (string): One of 'train', 'val', or 'test' to specify the dataset split.
            transform (callable, optional): Optional transform to be applied on a sample.
            low_res_size (tuple): The size for the low-resolution masks required by the trainer.
        """
        self.transform = transform
        self.low_res_size = low_res_size
        self.image_dir = os.path.join(base_dir, f'{split}_images')
        self.mask_dir = os.path.join(base_dir, f'{split}_masks')

        # Get all image filenames, assuming they are .png
        self.image_filenames = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])

        # Low-resolution mask transform
        self.low_res_transform = transforms.Compose([
            transforms.Resize(self.low_res_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])


    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        # Assume mask has the same name as the image
        mask_path = os.path.join(self.mask_dir, img_name)

        # Open image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") # Convert to single channel grayscale

        sample = {'image': image, 'label': mask}

        if self.transform:
            sample = self.transform(sample)

        # Create the low-resolution mask required by the trainer
        # We need to do this after the main transform in case of augmentations like rotation
        low_res_label_tensor = self.low_res_transform(Image.fromarray(sample['label'].numpy().astype(np.uint8)))
        sample['low_res_label'] = low_res_label_tensor.squeeze(0)

        return sample


# This RandomGenerator is a simplified version of what might have been in the original file.
# It handles resizing, random flipping, and conversion to tensors.
class RandomGenerator(object):
    def __init__(self, output_size=(224, 224)):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # Random horizontal flip
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # Random vertical flip
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            label = label.transpose(Image.FLIP_TOP_BOTTOM)

        # Convert to numpy arrays
        image = np.array(image, dtype=np.float32) / 255.0
        label = np.array(label, dtype=np.uint8)

        # Resize
        x, y = image.shape[:2]
        if x != self.output_size[0] or y != self.output_size[1]:
            image = self.resize_image(image, self.output_size)
            label = self.resize_image(label, self.output_size, interpolation=Image.NEAREST)

        # Normalize and convert to tensor
        image = torch.from_numpy(image.copy()).permute(2, 0, 1) # HWC to CHW
        label = torch.from_numpy(label.copy()).long()

        return {'image': image, 'label': label}

    def resize_image(self, image, output_size, interpolation=Image.BILINEAR):
        img_pil = Image.fromarray(image.astype(np.uint8))
        resized_img_pil = img_pil.resize(output_size, interpolation)
        return np.array(resized_img_pil)