import argparse
import json
import logging
import os
from pathlib import Path
import copy
from datetime import datetime
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
import wandb
from dotenv import load_dotenv

from models import ImageClassifier
from data import get_dataloader

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# Torch setup
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(0)

# Load .env (for WandB API key)
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)


def model_fn(model_dir):
    """Load model from file for SageMaker"""
    logger.info("Loading the model.")
    model = ImageClassifier()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


def save_model(model, model_dir):
    """Save model to file for SageMaker"""
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    # Recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)

def save_checkpoint(checkpoint_dir, run_name=None, checkpoint_name=None, model=None, optimizer=None, epoch=0, loss=None, args=None):
    """Save checkpoint"""
    logger.info("Saving checkpoint...")

    # Get checkpoint path
    run_checkpoint_dir = Path.cwd() / checkpoint_dir / run_name
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_checkpoint_dir / checkpoint_name
    config_path = run_checkpoint_dir / "config.json"

    checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': args
            }

    with open(config_path, 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4, separators=(',', ': '))

    # Recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint, model, optimizer):
    """Load checkpoint
    checkpoint is the path of the checkpoint
    """
    logger.info("Loading checkpoint")
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


def main(args):
    """Train and evaluate a ResNet18 car model classifier"""
    wandb.init(config=args, project=args.project)
    model = ImageClassifier(
        feature_extractor_name=args.feature_extractor, freeze_feature_extractor=args.freeze_feature_extractor, pretrained=True, num_classes=args.num_classes
    )
    model = model.to(device)
    summary(model, input_size=(3, args.input_size, args.input_size))

    # Normalize as described in https://pytorch.org/docs/stable/torchvision/models.html
    # The pretrained model expects input images normalized in this way
    # The values have been calculated on a random subset of ImageNet training images
    # The exact subset has been lost
    # See: https://github.com/pytorch/vision/issues/1439
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    data_transforms = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    dataloader_source_train = None
    if args.data_dir_train_source is not None:
        dataloader_source_train = get_dataloader(args.data_dir_train_source, data_transforms, args.batch_size, weighted_sampling=True, num_workers=args.workers)
    
    dataloader_val = None
    if args.data_dir_val is not None:
        dataloader_val = get_dataloader(args.data_dir_val, data_transforms, args.batch_size_test)


    dataloader_target_train = None
    if args.data_dir_train_target:
        if args.mode == 'dann':
            num_samples = len(dataloader_source_train.dataset)
        elif args.mode == 'source':
            num_samples = None
        dataloader_target_train = get_dataloader(args.data_dir_train_target, data_transforms, args.batch_size, weighted_sampling=True, num_samples=num_samples)


    # TODO: Do I need to check whether param.requires_grad == True?
    optimizer = optim.SGD(
        [param for param in model.parameters() if param.requires_grad == True],
        lr=args.lr,
        momentum=args.momentum,
    )

    now = datetime.now()
    args.run_name = f"{now.timestamp()}-{now.strftime('%Y-%m-%d-%H-%M-%S')}-{args.run_name}"

    # Load checkpoint
    completed_epochs = 0
    if args.checkpoint is not None:
        model, optimizer, completed_epochs, loss = load_checkpoint(args.checkpoint, model, optimizer)

    if dataloader_source_train is not None:
        train(
            model,
            dataloader_source_train,
            dataloader_target_train,
            dataloader_val,
            optimizer,
            completed_epochs,
            args,
        )

    if dataloader_source_train is None and dataloader_val is not None:
        acc = test(model, dataloader_val)


def train(
    model=None, data_loader_source=None, data_loader_target=None, data_loader_val=None, optimizer=None, completed_epochs=None, args=None
):
    """Train a model with a ResNet18 feature extractor on data from the source and target domain, adapted from: https://github.com/fungtion/DANN_py3/blob/master/main.py"""

    best_acc = 0
    best_epoch_loss_label_src = sys.float_info.max
    val_acc_history = []

    for epoch in range(completed_epochs + 1, completed_epochs + args.epochs + 1):
        len_dataloader = len(data_loader_source)
        if data_loader_target is not None and args.mode == "dann":
            len_dataloader = min(len(data_loader_source), len(data_loader_target))
            data_target_iter = iter(data_loader_target)
        data_source_iter = iter(data_loader_source)
        loss_label_classifier = torch.nn.CrossEntropyLoss()
        loss_domain_classifier = torch.nn.CrossEntropyLoss()

        running_loss_label_src = 0

        model.train()
        for i in range(1, len_dataloader+1):

            # Training with source data
            data_source = data_source_iter.next()
            src_img, src_label = data_source
            src_img, src_label = src_img.to(device), src_label.to(device)
            model.zero_grad()
            src_domain = torch.zeros_like(src_label)
            src_domain = src_domain.to(device)
            class_output, domain_output = model(src_img)
            preds = class_output.argmax(dim=1)
            batch_acc_train = preds.eq(src_label).sum().item() / src_label.size(0)
            domain_preds = domain_output.argmax(dim=1)
            batch_acc_domain_src = domain_preds.eq(src_domain).sum().item() / src_domain.size(0)

            err_src_label = loss_label_classifier(class_output, src_label)
            running_loss_label_src += err_src_label.data.cpu().item()
            err_src_domain = loss_domain_classifier(domain_output, src_domain)

            # Training with target data
            err_tgt_domain = torch.FloatTensor(1).fill_(0)
            batch_acc_domain_tgt = 0
            if data_loader_target is not None and args.mode == "dann":
                data_target = data_target_iter.next()
                tgt_img, tgt_label = data_target
                tgt_img, tgt_label = tgt_img.to(device), tgt_label.to(device)
                tgt_domain = torch.ones_like(tgt_label)
                tgt_domain = tgt_domain.to(device)
                _, domain_output = model(tgt_img)
                domain_preds = domain_output.argmax(dim=1)
                batch_acc_domain_tgt = domain_preds.eq(tgt_domain).sum().item() / tgt_domain.size(0)
                err_tgt_domain = loss_domain_classifier(domain_output, tgt_domain)

            if args.mode == "dann":
                err = err_src_label + err_tgt_domain + err_src_domain
            else:
                err = err_src_label
            err.backward()
            optimizer.step()

            print(
                "epoch: %d, [iter: %d / all %d], err_src_label: %f, err_src_domain: %f, err_tgt_domain: %f, b_acc: %f, b_acc_domain_src: %f, b_acc_domain_tgt: %f"
                % (
                    epoch,
                    i,
                    len_dataloader,
                    err_src_label.data.cpu().item(),
                    err_src_domain.data.cpu().item(),
                    err_tgt_domain.data.cpu().item(),
                    batch_acc_train,
                    batch_acc_domain_src,
                    batch_acc_domain_tgt
                )
            )
            wandb.log(
                {
                    "loss_src_label": err_src_label.data.cpu().item(),
                    "loss_src_domain": err_src_domain.data.cpu().item(),
                    "loss_tgt_domain": err_tgt_domain.data.cpu().item(),
                    "batch_acc_train": batch_acc_train,
                    "batch_acc_domain_src": batch_acc_domain_src,
                    "batch_acc_domain_tgt": batch_acc_domain_tgt
                }
            )
        
        epoch_loss_label_src = running_loss_label_src / len_dataloader
        print(
            "epoch: %d, err_src_label: %f" % (epoch, epoch_loss_label_src)
        )
        save_checkpoint(checkpoint_dir=args.checkpoint_dir, run_name=args.run_name, checkpoint_name="latest.pt", model=model, optimizer=optimizer, epoch=epoch, args=args)

        if data_loader_val is not None:
            acc = test(model, data_loader_val)
            val_acc_history.append(acc)

        if epoch_loss_label_src < best_epoch_loss_label_src:
            best_epoch_loss_label_src = epoch_loss_label_src
            save_checkpoint(checkpoint_dir=args.checkpoint_dir, run_name=args.run_name, checkpoint_name="best.pt", model=model, epoch=epoch, optimizer=optimizer, args=args)

    return model, val_acc_history


def test(model, data_loader):
    """Test model"""
    model.eval()
    loss_label_classifier = torch.nn.CrossEntropyLoss(reduction="sum")
    test_loss = 0
    correct = 0
    preds = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            class_output, _ = model(data)
            test_loss += loss_label_classifier(class_output, target).item()
            pred = class_output.argmax(dim=1)
            preds.extend(pred.tolist())
            correct += pred.eq(target).sum().item()
    test_loss /= len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)
    print(
        "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.4f})\n".format(
            test_loss, correct, len(data_loader.dataset), acc,
        )
    )
    wandb.log({"loss_tgt_label_val": test_loss, "acc_val": acc})

    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        metavar="E",
        help="What is this run called?",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        metavar="N",
        help="What dimension is the input? (default: 224)",
    )
    parser.add_argument(
        "--feature-extractor",
        type=str,
        default="resnet18",
        metavar="N",
        help="Which feature extractor to use? (default: resnet18)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
        metavar="N",
        help="How many classes are there? (default: 10)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="dann",
        metavar="M",
        help="""Which mode? dann: Train on source and target domain, evaluate on target domain,
            source: train on source domain, evaluate on source domain
            (evaluate on target domain, if --data-dir-target-domain passed) (default: dann)""",
    )
    parser.add_argument(
        "--freeze-feature-extractor",
        action='store_true',
        default=False,
        help="""Freeze feature extractor""",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--batch-size-test",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=0.8,
        metavar="N",
        help="train size (default: 0.8)",
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
        "--project",
        type=str,
        default="unsupervised-domain-adaptation",
        metavar="P",
        help="WandB project name",
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
        "--checkpoint-dir",
        type=str,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
    )
    parser.add_argument(
        "--data-dir-train-source",
        type=str,
        default=os.environ["SM_CHANNEL_SOURCE_TRAIN"]
        if "SM_CHANNEL_SOURCE_TRAIN" in os.environ
        else None,
    )
    parser.add_argument(
        "--data-dir-train-target",
        type=str,
        default=os.environ["SM_CHANNEL_TARGET_TRAIN"]
        if "SM_CHANNEL_TARGET_TRAIN" in os.environ
        else None,
    )
    parser.add_argument(
        "--data-dir-val",
        type=str,
        default=os.environ["SM_CHANNEL_VAL"]
        if "SM_CHANNEL_VAL" in os.environ
        else None,
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=os.environ["SM_NUM_GPUS"] if "SM_NUM_GPUS" in os.environ else None,
    )
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
        help='number of data loading workers (default: 4)')

    main(parser.parse_args())
