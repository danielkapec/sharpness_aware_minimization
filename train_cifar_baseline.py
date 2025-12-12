import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models
from pathlib import Path

import time
import argparse

import json
import csv
import os
from datetime import datetime

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# print("Using device:", device)

def get_device():
    
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
def get_dataloaders(batch_size=128, num_workers=2, train_fraction=1.0):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    data_root = Path("./data")

    full_train_dataset = datasets.CIFAR10(
        root = data_root,
        train = True,
        download = True,
        transform = transform_train,
    )

    if train_fraction < 1.0:
        n_total = len(full_train_dataset)
        n_sub = int(train_fraction*n_total)
        indices = torch.randperm(n_total)[:n_sub]
        train_dataset = Subset(full_train_dataset, indices)
        print(f"Using subset of cifar-10: {n_sub}/{n_total} train samples")
    else:
        train_dataset = full_train_dataset





    test_dataset = datasets.CIFAR10(
        root = data_root,
        train = False,
        download = True,
        transform = transform_test,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, test_loader


def get_model(num_classes=10):
    #resnet-18 adapted for cifar10
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )

    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model


def train_one_epoch(model, optimizer, dataloader, device, criterion):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, targets in dataloader:
        images = images.to(device, non_blocking = True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        running_correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc



@torch.no_grad()
def evaluate(model, dataloader, device, criterion):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, targets in dataloader:
        images = images.to(device, non_blocking = True)
        targets = targets.to(device, non_blocking = True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item()*images.size(0)
        _, preds = outputs.max(1)
        running_correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", "-e",
        type = int,
        default = 100,
        help = "Number of training epochs",
    )
    parser.add_argument(
        "--num-workers", "-nw",
        type = int,
        default = 0,
        help = "Number of workers for data loader"
    )
    return parser.parse_args()


def main(num_epochs: int, num_workers: int):

    #log the configuration for the run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"experiments/run_{timestamp}")
    run_dir.mkdir(parents = True, exist_ok = True)

    config = {
        "num_epochs": num_epochs,
        "num_workers": num_workers,
        "train_fraction": .2,
        "batch_size": 128,
        "base_lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "architecture": "resnet18_cifar10",
        "device": "auto" 
    }

    with open(run_dir/"config.json","w") as f:
        json.dump(config, f, indent = 4)
    print(f"Run directory created at {run_dir}")


    device = get_device()
    # running 140s per epoch on mps, 6 min on cpu, 40s on gpu
    # device = torch.device("cpu")
    print("Using device:", device)
    config["device"] = str(device)





    train_loader, test_loader = get_dataloaders(batch_size=config["batch_size"], 
                                                num_workers=num_workers,
                                                train_fraction=config["train_fraction"],
                                                )

    model = get_model(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()


    optimizer = optim.SGD(
        model.parameters(), 
        lr=config["base_lr"], 
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )

    best_acc = 0.0 # in case we overfit
    log_path = run_dir / "log.csv"
    with open(log_path, "w", newline = "") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_acc", 
            "test_loss", "test_acc", "lr", "time"
        ])




    for epoch in range(1, num_epochs + 1):
        start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model, optimizer, train_loader, device, criterion
        )

        test_loss, test_acc = evaluate(model, test_loader, device, criterion)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        elapsed = time.perf_counter() - start

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} | "
            f"time={elapsed:.1f}s"
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_loss, train_acc, 
                test_loss, test_acc, current_lr, elapsed
            ])

        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "config": config,
            "metrics": {"test_acc": test_acc, "test_loss": test_loss}
        }
        if test_acc > best_acc:
            best_acc = test_acc
            # Save to run_dir, NOT /training_checkpoints
            torch.save(checkpoint, run_dir / "model_best.pt") 
            print(f"   --> New best accuracy! Saved model_best.pt")


        # --- UPDATED: Periodic Save ---
        if epoch % 10 == 0:
            # FIX: Used run_dir instead of absolute path
            ckpt_path = run_dir / f"resnet_epoch_{epoch:03d}.pt"
            torch.save(checkpoint, ckpt_path)
            print(f"   --> Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args.epochs, args.num_workers)