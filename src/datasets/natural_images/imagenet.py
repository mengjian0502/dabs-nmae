import os

import PIL
from PIL import Image
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 1000000000

from torch.utils.data import Dataset
from torchvision import datasets, transforms

from src.datasets.specs import Input2dSpec


class ImageNet(Dataset):
    # Dataset information.
    NUM_CLASSES = 1000
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 3
    MAE_OUTPUT_SIZE = 768

    TRANSFORMS = transforms.Compose(
        [
            transforms.Resize(INPUT_SIZE),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
        ]
    )

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__()
        # self.root = os.path.join(base_root, 'natural_images', 'imagenet')
        # self.root = os.path.join(base_root, 'imagenet')
        self.root = base_root
        if not os.path.isdir(self.root):
            os.makedirs(self.root)

        if download and not os.path.exists(self.root):
            print('ImageNet not publicly available. Please edit self.root to point to your ImageNet path.')

        root = os.path.join(self.root, 'train' if train else 'val')
        self.dataset = datasets.ImageFolder(root=root)

    def __getitem__(self, index):
        img, label = self.dataset.samples[index]
        img = Image.open(img).convert(mode='RGB')
        img = self.TRANSFORMS(img)
        return index, img, label

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def num_classes():
        return ImageNet.NUM_CLASSES

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input2dSpec(input_size=ImageNet.INPUT_SIZE, patch_size=ImageNet.PATCH_SIZE, in_channels=ImageNet.IN_CHANNELS),
        ]
