import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path

import time

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# print("Using device:", device)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
def get_dataloaders(batch_size=128, num_workers=2):
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

    train_dataset = datasets.CIFAR10(
        root = data_root,
        train = True,
        download = True,
        transform = transform_train,
    )

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


def main():

    device = get_device()
    # running 140s per epoch on mps, want to test on cpu
    # device = torch.device("cpu")

    print("Using device:", device)

    train_loader, test_loader = get_dataloaders(batch_size=128)

    model = get_model(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()

    base_lr = .1

    optimizer = optim.SGD(
        model.parameters(), lr=base_lr, momentum=0.9,
        weight_decay=5e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100
    )

    num_epochs = 100

    for epoch in range(1, num_epochs + 1):
        start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model, optimizer, train_loader, device, criterion
        )
        test_loss, test_acc = evaluate(model, test_loader, device, criterion)
        scheduler.step()

        elapsed = time.perf_counter() - start
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} | "
            f"time={elapsed:.1f}s"
        )

if __name__ == "__main__":
    main()