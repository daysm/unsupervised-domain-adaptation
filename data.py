import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
from torch.utils.data import WeightedRandomSampler


class DaimlerImageFolder(datasets.ImageFolder):

    # Get original image without transformation to be able to see original data
    def get_sample(self, index):
        img = Image.open(self.samples[index][0])
        return img


def get_train_val_loaders(
    data_dir, data_transforms, train_size=0.8, batch_size_train=32, batch_size_val=1000
):
    # TODO: Add calculation for num_samples of target domain
    dataset = DaimlerImageFolder(root=data_dir, transform=data_transforms)

    # Split into train and val
    len_train, len_val = (
        int(np.floor(len(dataset) * train_size)),
        int(np.ceil(len(dataset) * (1 - train_size))),
    )
    dataset_train, dataset_val = torch.utils.data.random_split(
        dataset, [len_train, len_val], generator=torch.Generator().manual_seed(0)
    )

    # Count class frequencies in training set
    dataset_train_labels = [dataset.targets[i] for i in dataset_train.indices]
    class_count_train = {
        target: dataset_train_labels.count(target)
        for target in set(dataset_train_labels)
    }
    class_count_train = [
        class_count_train[class_idx] for class_idx in sorted(class_count_train.keys())
    ]

    # One weight for each class
    weights = 1.0 / torch.tensor(class_count_train).float()
    
    # Convert to to make compatible with weights
    dataset_train_labels = torch.LongTensor(dataset_train_labels)
    
    # One weight for each sample, based on the sample's label 
    sample_weights = weights[dataset_train_labels]

    num_samples = len(dataset_train)
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples, generator=torch.Generator().manual_seed(0)
    )
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size_train, sampler=sampler
    )
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size_val)

    return dataloader_train, dataloader_val


if __name__ == "__main__":
    input_size = 224
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    data_transforms = transforms.Compose(
        [transforms.Resize((input_size, input_size)), transforms.ToTensor(), normalize]
    )

    dataloader_synth_train, dataloader_synth_val = get_train_val_loaders(
        "data/synthetic_new_for_model_classification_fixed_cropped",
        transforms,
        train_size=0.8,
    )
    dataloader_dealer_train, dataloader_dealer_val = get_train_val_loaders(
        "data/real_new_for_model_classification_cropped_cleaned",
        transforms,
        train_size=0.8,
    )
