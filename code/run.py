import argparse
import json
import logging
import os
import json
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
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from dotenv import load_dotenv

from models import ImageClassifier
from data import get_dataloader
from checkpoint import model_fn, save_model, load_checkpoint, save_checkpoint

# Use GPU, if available
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Load .env (for WandB API key)
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)


def main(args):
    """Train and evaluate a ResNet18 car model classifier"""
    torch.manual_seed(args.seed)

    model = ImageClassifier(
        feature_extractor_name=args.feature_extractor,
        freeze_feature_extractor=args.feature_extractor_weights == "freeze",
        pretrained=True,
        num_classes=args.num_classes,
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

    dataloader_primary_train = None
    if args.data_dir_train_primary:
        dataloader_primary_train = get_dataloader(
            data_dir=args.data_dir_train_primary,
            data_transforms=data_transforms,
            batch_size=args.batch_size,
            weighted_sampling=True,
            num_workers=args.workers,
        )

    dataloader_val = None
    if args.data_dir_val is not None:
        dataloader_val = get_dataloader(
            args.data_dir_val, data_transforms, args.batch_size_test
        )

    dataloader_aux_train = None
    if args.data_dir_train_aux:
        if args.mode == "dann":
            num_samples = len(dataloader_primary_train.dataset)
        elif args.mode == "source":
            num_samples = None
        dataloader_aux_train = get_dataloader(
            args.data_dir_train_aux,
            data_transforms,
            args.batch_size,
            weighted_sampling=True,
            num_samples=num_samples,
        )

    # TODO: Do I need to check whether param.requires_grad == True?
    optimizer = optim.SGD(
        [param for param in model.parameters() if param.requires_grad == True],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    now = datetime.now()
    args.run_name = (
        f"{now.timestamp()}-{now.strftime('%Y-%m-%d-%H-%M-%S')}-{args.run_name}"
    )

    # Load checkpoint
    completed_epochs = 0
    loss = None
    if args.checkpoint is not None:
        model, optimizer, completed_epochs, loss = load_checkpoint(
            args.checkpoint, model, optimizer
        )

    if dataloader_primary_train is not None:
        train(
            model=model,
            dataloader_primary=dataloader_primary_train,
            dataloader_aux=dataloader_aux_train,
            dataloader_val=dataloader_val,
            optimizer=optimizer,
            completed_epochs=completed_epochs,
            loss=loss,
            args=args,
        )

    if dataloader_primary_train is None and dataloader_val is not None:
        test_loss, acc = test(model, dataloader_val)



def train(
    model=None,
    dataloader_primary=None,
    dataloader_aux=None,
    dataloader_val=None,
    optimizer=None,
    completed_epochs=None,
    loss=None,
    args=None,
):
    """Train a model with a ResNet18 feature extractor on data from the primary and auxiliary domain(s), adapted from: https://github.com/fungtion/DANN_py3/blob/master/main.py"""
    wandb.init(config=args, project=args.project, name=args.run_name)
    best_acc = 0.0
    best_epoch_loss_label_primary = sys.float_info.max if loss is None else loss
    val_acc_history = []

    for epoch in range(completed_epochs + 1, completed_epochs + args.epochs + 1):
        len_dataloader = len(dataloader_primary)
        if dataloader_aux is not None and args.mode == "dann":
            len_dataloader = min(len(dataloader_primary), len(dataloader_aux))
            data_aux_iter = iter(dataloader_aux)
        data_primary_iter = iter(dataloader_primary)
        loss_label_classifier = torch.nn.CrossEntropyLoss()
        loss_domain_classifier = torch.nn.CrossEntropyLoss()

        running_loss_label_primary = 0

        model.train()
        for i in range(1, len_dataloader + 1):
            # Training with primary data
            data_primary = data_primary_iter.next()
            primary_img, primary_label = data_primary
            primary_img, primary_label = (
                primary_img.to(device),
                primary_label.to(device),
            )
            model.zero_grad()
            primary_domain = torch.zeros_like(primary_label)
            primary_domain = primary_domain.to(device)
            class_output, domain_output = model(primary_img)
            preds = class_output.argmax(dim=1)
            batch_acc_train = preds.eq(primary_label).sum().item() / primary_label.size(
                0
            )
            domain_preds = domain_output.argmax(dim=1)
            batch_acc_domain_primary = domain_preds.eq(
                primary_domain
            ).sum().item() / primary_domain.size(0)

            loss_primary_label = loss_label_classifier(class_output, primary_label)
            running_loss_label_primary += loss_primary_label.data.cpu().item()
            loss_primary_domain = loss_domain_classifier(domain_output, primary_domain)

            # Training with auxiliary data
            loss_aux_domain = torch.FloatTensor(1).fill_(0)
            batch_acc_domain_aux = 0
            if dataloader_aux is not None and args.mode == "dann":
                data_aux = data_aux_iter.next()
                aux_img, aux_label = data_aux
                aux_img, aux_label = aux_img.to(device), aux_label.to(device)
                aux_domain = torch.ones_like(aux_label)
                aux_domain = aux_domain.to(device)
                _, domain_output = model(aux_img)
                domain_preds = domain_output.argmax(dim=1)
                batch_acc_domain_aux = domain_preds.eq(
                    aux_domain
                ).sum().item() / aux_domain.size(0)
                loss_aux_domain = loss_domain_classifier(domain_output, aux_domain)

            if args.mode == "dann":
                loss = loss_primary_label + loss_aux_domain + loss_primary_domain
            else:
                loss = loss_primary_label
            loss.backward()
            optimizer.step()

            if i % args.log_interval == 0:
                print(
                    "epoch: %d, [iter: %d / all %d], loss_primary_label: %f, loss_primary_domain: %f, loss_aux_domain: %f, acc_primary_label_batch: %f, acc_primary_domain_batch: %f, acc_aux_domain_batch: %f"
                    % (
                        epoch,
                        i,
                        len_dataloader,
                        loss_primary_label.data.cpu().item(),
                        loss_primary_domain.data.cpu().item(),
                        loss_aux_domain.data.cpu().item(),
                        batch_acc_train,
                        batch_acc_domain_primary,
                        batch_acc_domain_aux,
                    )
                )
            wandb.log(
                {
                    "loss_primary_label": loss_primary_label.data.cpu().item(),
                    "loss_primary_domain": loss_primary_domain.data.cpu().item(),
                    "loss_aux_domain": loss_aux_domain.data.cpu().item(),
                    "acc_primary_label_batch": batch_acc_train,
                    "acc_primary_domain_batch": batch_acc_domain_primary,
                    "acc_aux_domain_batch": batch_acc_domain_aux,
                }
            )

        epoch_loss_label_primary = running_loss_label_primary / len_dataloader
        print("epoch: %d, loss_primary_label: %f" % (epoch, epoch_loss_label_primary))

        if dataloader_val is not None:
            val_loss, val_acc = test(model, dataloader_val)
            print(
                "val_loss: %f, val_acc: %f"
                % (
                    val_loss, val_acc
                )
            )
            wandb.log(
                {
                    "val_loss_label": val_loss,
                    "val_acc_label": val_acc,
                }
            )
            val_acc_history.append(val_acc)
            if val_acc > best_acc:
                best_acc = val_acc

        if epoch_loss_label_primary < best_epoch_loss_label_primary:
            best_epoch_loss_label_primary = epoch_loss_label_primary
            save_checkpoint(
                checkpoint_dir=args.checkpoint_dir,
                run_name=args.run_name,
                checkpoint_name="best.pt",
                model=model,
                epoch=epoch,
                loss=best_epoch_loss_label_primary,
                optimizer=optimizer,
                args=args,
            )
        save_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            run_name=args.run_name,
            checkpoint_name="latest.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=best_epoch_loss_label_primary,
            args=args,
        )

    return model, val_acc_history


def test(model, dataloader):
    """Test model"""
    model.eval()
    loss_label_classifier = torch.nn.CrossEntropyLoss(reduction="sum")
    test_loss = 0
    correct = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for i, (data, label) in enumerate(dataloader):
            print(i)
            data, label = data.to(device), label.to(device)
            class_output, _ = model(data)
            test_loss += loss_label_classifier(class_output, label).item()
            preds = class_output.argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(label.tolist())
            correct += preds.eq(label).sum().item()
    test_loss /= len(dataloader.dataset)
    acc = correct / len(dataloader.dataset)
    print(
        "val_loss_label: %f,  val_acc_label: %f"
        % (
            test_loss,
            acc
        )
    )
    class_names = list(dataloader.dataset.class_to_idx.keys())
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    report_text = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0 )
    print(report_text)
    print(report_dict)
    return test_loss, acc


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
        "--num-classes",
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
        help="""Which mode? dann: Train on primary and auxilary domain(s),
            source: train on primary domain (default: dann)""",
    )
    parser.add_argument(
        "--feature-extractor-weights",
        type=str,
        default="freeze",
        help="""Freeze or train (finetune) feature extractor""",
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
        default=0.01,
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
        "--weight-decay",
        type=float,
        default=0.001,
        metavar="L2",
        help="L2 regularization (default: 0.9)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
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
        "--checkpoint-dir", type=str,
    )
    parser.add_argument(
        "--checkpoint", type=str,
    )
    parser.add_argument(
        "--data-dir-train-primary",
        type=str,
        # action="append",
        default=os.environ["SM_CHANNEL_TRAIN_PRIMARY"]
        if "SM_CHANNEL_TRAIN_PRIMARY" in os.environ
        else None,
    )
    parser.add_argument(
        "--data-dir-train-aux",
        type=str,
        # action="append",
        default=os.environ["SM_CHANNEL_TRAIN_AUX"]
        if "SM_CHANNEL_TRAIN_AUX" in os.environ
        else None,
    )
    parser.add_argument(
        "--data-dir-val",
        # action="append",
        type=str,
        default=os.environ["SM_CHANNEL_VAL"]
        if "SM_CHANNEL_VAL" in os.environ
        else None,
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )

    main(parser.parse_args())

    # class_names = ['1179', '1569', '1671', '1770', '2050', '2054', '2130', '2470', '2533', '2539']
    # all_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
    # all_preds = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 3, 6, 6, 0, 3, 3, 3, 3, 7, 1, 1, 9, 2, 2, 2, 7, 2, 2, 2, 0, 3, 3, 7, 0, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 7, 3, 3, 7, 1, 3, 3, 3, 3, 3, 7, 3, 3, 3, 3, 3, 3, 7, 3, 0, 7, 7, 0, 4, 4, 5, 6, 5, 0, 0, 5, 5, 4, 4, 7, 4, 4, 4, 1, 7, 6, 4, 4, 4, 4, 4, 6, 6, 6, 6, 3, 0, 4, 4, 4, 7, 7, 7, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 3, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 9, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 8, 8, 8, 7, 0, 9, 9, 3, 9, 9, 8, 9, 9, 9, 9, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 2, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0, 9, 9, 9, 9, 9, 9, 7, 9, 9]
    # cf_matrix = confusion_matrix(all_labels, all_preds)
    # print(cf_matrix)
    # report_text = classification_report(all_labels, all_preds, labels=class_names, target_names=class_names)
    # report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    # print(report_text)
    # print(report_dict)