import os
import torch
from torchvision import datasets, transforms
from .autoaugment import ImageNetPolicy


def build_train_valid_datasets(data_path, crop_size=224, color_jitter=True):

    #training dataset
    train_data_path = os.path.join(data_path[0], "train")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.5, 0.5, 0.5])
    process = [
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
    ]
    if color_jitter:
        process += [
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            )
        ]
    fp16_t = transforms.ConvertImageDtype(torch.half)
    process += [ImageNetPolicy(), transforms.ToTensor(), normalize, fp16_t]
    transform_train = transforms.Compose(process)
    train_data = datasets.ImageFolder(
        root=train_data_path, transfor=transform_train
    )

    #validation dataset
    valid_data_path = os.path.join(data_path[0], "val")
    transform_val = transforms.Compose(
        [
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
            fp16_t
        ]
    )
    val_Data = datasets.ImageFolder(
        root=val_data_path, transform=transform_val
    )

    return train_data, val_data
