import os
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class Pix2PixDataset(Dataset):
    def __init__(self, root, invert=False, transform=None, device=None):
        super().__init__()
        self.root = root
        self.invert = invert
        self.transform = transform
        self.device = device
        self.filenames = sorted(os.listdir(root))

    def __len__(self):
        return len(self.filenames)

    def get_image_pair(self, filepath):
        raise NotImplementedError(f"Pix2PixDataset is an Abstract Base Class. Use an actual Dataset object.")

    def __getitem__(self, index):
        filepath = os.path.join(self.root, self.filenames[index])
        image_a, image_b = self.get_image_pair(filepath)
        if self.invert:
            image_b, image_a = image_a, image_b

        if self.transform is not None:
            seed = random.randint(0, 2**32)
            random.seed(seed)
            image_a = self.transform(image_a)
            random.seed(seed)
            image_b = self.transform(image_b)

        if self.device is not None:
            image_a = image_a.to(self.device)
            image_b = image_b.to(self.device)

        return image_a, image_b

class MapsDataset(Pix2PixDataset):
    def get_image_pair(self, filepath):
        image = Image.open(filepath).convert('RGB')
        width, height = image.size
        image_a = image.crop((0, 0, width // 2, height))
        image_b = image.crop((width // 2, 0, width, height))
        return image_a, image_b

class PlacesDataset(Pix2PixDataset):
    def get_image_pair(self, filepath):
        image_b = Image.open(filepath).convert('RGB')
        image_a = image_b.convert('L')
        return image_a, image_b

class CityscapesDataset(Pix2PixDataset):
    def get_image_pair(self, filepath):
        image = Image.open(filepath).convert('RGB')
        width, height = image.size
        image_b = image.crop((0, 0, width // 2, height))
        image_a = image.crop((width // 2, 0, width, height))
        return image_a, image_b

class FacadesDataset(Pix2PixDataset):
    def get_image_pair(self, filepath):
        image = Image.open(filepath).convert('RGB')
        width, height = image.size
        image_b = image.crop((0, 0, width // 2, height))
        image_a = image.crop((width // 2, 0, width, height))
        return image_a, image_b

def dataloader(root, dataset, invert=False, device=None, train=True, resize=None, crop=None,
                    batch_size=1, shuffle=False):
    
    if dataset == 'places':
        mean, std = (0.5, ), (0.5, )
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    if train:
        transform = transforms.Compose([
            transforms.Resize(resize, Image.BICUBIC),
            transforms.RandomCrop(crop),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(crop, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    if dataset == 'places':
        pix2pix_dataset = PlacesDataset(root, invert, transform=transform, device=device)
    elif dataset == 'maps':
        pix2pix_dataset = MapsDataset(root, invert, transform=transform, device=device)
    elif dataset == 'cityscapes':
        pix2pix_dataset = CityscapesDataset(root, invert, transform=transform, device=device)
    elif dataset == 'facades':
        pix2pix_dataset = FacadesDataset(root, invert, transform=transform, device=device)
    else:
        raise AssertionError(f"Dataset {dataset} not found. Use one among 'cityscapes | facades | maps | places'")

    dataloader = DataLoader(dataset=pix2pix_dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
