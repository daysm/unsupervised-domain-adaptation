import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from PIL import Image
from torchvision import datasets, models, transforms
from torch.utils.data import WeightedRandomSampler


class DaimlerImageFolder(datasets.ImageFolder):
    """A wrapper around ImageFolder to allow access to the raw data without transforms"""

    # Get original image without transformation to be able to see original data
    def get_sample(self, index):
        img = Image.open(self.samples[index][0])
        return img


def get_dataloader(
    data_dir,
    data_transforms,
    batch_size=32,
    weighted_sampling=False,
    num_samples=None,
    num_workers=4,
):
    """Get dataloaders for training and validation"""
    dataset = DaimlerImageFolder(root=data_dir, transform=data_transforms)

    if weighted_sampling:
        sampler = get_weighted_random_sampler(dataset, num_samples=num_samples)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
        )

    return dataloader


def get_weighted_random_sampler(dataset, num_samples=None):
    # Count class frequencies
    class_counts = get_class_counts(dataset)
    sample_weights = get_sample_weights(dataset.targets, class_counts)

    # If the target domain dataset is much smaller we can pass the number of
    # training samples to sample directly
    if num_samples is None:
        num_samples = len(dataset)

    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples, generator=torch.Generator().manual_seed(0)
    )

    return sampler


def get_class_counts(dataset):
    class_counts = {
        target: dataset.targets.count(target) for target in set(dataset.targets)
    }
    class_counts = [
        class_counts[class_idx] for class_idx in sorted(class_counts.keys())
    ]
    return class_counts


def get_sample_weights(dataset_labels, class_counts):
    # One weight for each class
    weights = 1.0 / torch.tensor(class_counts).float()

    # Convert to to make compatible with weights
    dataset_labels = torch.LongTensor(dataset_labels)

    # One weight for each sample, based on the sample's label
    sample_weights = weights[dataset_labels]

    return sample_weights


if __name__ == "__main__":
    input_size = 224
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    data_transforms = transforms.Compose(
        [transforms.Resize((input_size, input_size)), transforms.ToTensor(), normalize]
    )

    dataloader = get_dataloader(
        [
            "data/real_new_for_model_classification_cropped_cleaned_test_set",
            "data/real_new_for_model_classification_cropped_cleaned_training_set",
        ],
        data_transforms,
        batch_size=32,
        weighted_sampling=True,
    )
