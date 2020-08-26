import argparse
import json
import logging
import os
import copy
import sys
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms
from torchsummary import summary

from models import ResNet18
from data import get_train_val_loaders

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# Torch setup
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(0)


def model_fn(model_dir):
    model = torch.nn.DataParallel(ResNet18())
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    # Recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)


def train(args):
    model = ResNet18(dann=args.dann, freeze_feature_extractor=False)
    model = model.to(device)

    input_size = 224
    summary(model, input_size=(3, input_size, input_size))

    # Normalize as described in https://pytorch.org/docs/stable/torchvision/models.html
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    data_transforms = transforms.Compose(
        [transforms.Resize((input_size, input_size)), transforms.ToTensor(), normalize]
    )

    dataloader_synth_train, dataloader_synth_val = get_train_val_loaders(
        args.source_domain_dir, data_transforms, train_size=0.8
    )
    dataloader_dealer_train, dataloader_dealer_val = get_train_val_loaders(
        args.target_domain_dir, data_transforms, train_size=0.8
    )

    # TODO: Do I need to check whether param.requires_grad == True?
    optimizer = optim.SGD(
        [param for param in model.parameters() if param.requires_grad == True],
        lr=args.lr,
        momentum=args.momentum,
    )

    if model.dann:
        train_dann(
            model,
            dataloader_synth_train,
            dataloader_dealer_train,
            optimizer,
            args.epochs,
        )
        test_dann(model, dataloader_dealer_val)
    else:
        criterion = nn.CrossEntropyLoss()
        dataloaders = {"train": dataloader_synth_train, "val": dataloader_synth_val}
        train_source(model, dataloaders, criterion, optimizer, args.epochs)


def train_source(
    model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False
):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == "train":
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def train_dann(model, data_loader_source, data_loader_target, optimizer, num_epochs):
    for epoch in range(1, num_epochs + 1):
        len_dataloader = min(len(data_loader_source), len(data_loader_target))
        data_source_iter = iter(data_loader_source)
        data_target_iter = iter(data_loader_target)
        loss_classifier = torch.nn.CrossEntropyLoss()
        loss_discriminator = torch.nn.CrossEntropyLoss()

        model.train()
        for i in range(len_dataloader):

            # Training with source data
            data_source = data_source_iter.next()
            src_img, src_label = data_source
            src_img, src_label = src_img.to(device), src_label.to(device)
            model.zero_grad()
            src_domain = torch.zeros_like(src_label)
            src_domain = src_domain.to(device)
            class_output, domain_output = model(src_img)
            err_src_label = loss_classifier(class_output, src_label)
            err_src_domain = loss_discriminator(domain_output, src_domain)

            # Training with target data
            data_target = data_target_iter.next()
            tgt_img, tgt_label = data_target
            tgt_img, tgt_label = tgt_img.to(device), tgt_label.to(device)
            tgt_domain = torch.ones_like(tgt_label)
            tgt_domain = tgt_domain.to(device)
            _, domain_output = model(tgt_img)
            err_tgt_domain = loss_discriminator(domain_output, tgt_domain)

            err = err_src_label + err_tgt_domain + err_src_domain
            err.backward()
            optimizer.step()

            print(
                "epoch: %d, [iter: %d / all %d], err_src_label: %f, err_src_domain: %f, err_tgt_domain: %f"
                % (
                    epoch,
                    i,
                    len_dataloader,
                    err_src_label.data.cpu().item(),
                    err_src_domain.data.cpu().item(),
                    err_tgt_domain.data.cpu().item(),
                )
            )


def test_dann(model, data_loader):
    model.eval()
    loss_classifier = torch.nn.NLLLoss(reduction="sum")
    test_loss = 0
    correct = 0
    preds = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            class_output, _ = model(data)
            test_loss += loss_classifier(class_output, target).item()
            pred = class_output.argmax(dim=1)
            preds.extend(pred.tolist())
            correct += pred.eq(target).sum()
    test_loss /= len(data_loader.dataset)
    print(
        "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(data_loader.dataset),
            100.0 * correct / len(data_loader.dataset),
        )
    )

    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--mode",
        type=str,
        default='dann',
        metavar="M",
        help="Which mode? dann, source (default: dann)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status (default: 100)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu, default: none)",
    )

    # Container environment
    parser.add_argument(
        "--hosts",
        type=list,
        default=json.loads(os.environ["SM_HOSTS"])
        if "SM_HOSTS" in os.environ
        else None,
    )
    parser.add_argument(
        "--current-host",
        type=str,
        default=os.environ["SM_CURRENT_HOST"]
        if "SM_CURRENT_HOST" in os.environ
        else None,
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ["SM_MODEL_DIR"] if "SM_MODEL_DIR" in os.environ else None,
    )
    parser.add_argument(
        "--source-domain-dir",
        type=str,
        default=os.environ["SM_CHANNEL_SOURCE"]
        if "SM_CHANNEL_SOURCE" in os.environ
        else None,
    )
    parser.add_argument(
        "--target-domain-dir",
        type=str,
        default=os.environ["SM_CHANNEL_TARGET"]
        if "SM_CHANNEL_TARGET" in os.environ
        else None,
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=os.environ["SM_NUM_GPUS"] if "SM_NUM_GPUS" in os.environ else None,
    )

    train(parser.parse_args())
