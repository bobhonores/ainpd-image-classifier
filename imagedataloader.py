import torch
from torchvision import datasets, transforms


def create_dataloaders(image_data_directory):
    data_transforms = {'train': transforms.Compose([transforms.RandomRotation(30),
                                                    transforms.RandomResizedCrop(
                                                        224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])]),
                       'valid': transforms.Compose([transforms.Resize(224),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])])}
    image_datasets = {
        'train': datasets.ImageFolder(image_data_directory + '/train', transform=data_transforms['train']),
        'valid': datasets.ImageFolder(image_data_directory + '/valid', transform=data_transforms['valid'])}
    image_dataloders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
                        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32)}

    return image_datasets, image_dataloders
