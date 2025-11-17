# tiny_image_classifier.py
import os
import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import prune

import torchvision
import torchvision.transforms as T

from tqdm import tqdm


# ==========================
#  Ustawienia / hiperparametry
# ==========================
BATCH_SIZE = 128
BASE_EPOCHS = 50         # liczba epok dla modelu bazowego (np. 80–100 na finalne wyniki)
FINETUNE_EPOCHS = 20     # liczba epok po pruning
BASE_LR = 1e-3
FINETUNE_LR = 5e-4
PRUNE_AMOUNT = 0.6       # procent wag do wycięcia (0.6 -> 60%)
DATA_DIR = "./data"
SAVE_DIR = "./models"
NUM_WORKERS = 4
SEED = 42
USE_PRETRAINED = True    # czy używać ImageNet pretrain do MobileNetV2


# ==========================
#  Utils
# ==========================
def set_seed(seed: int = 42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_model_size_mb(model: nn.Module):
    # Zakładamy float32 -> 4 bajty na parametr
    _, trainable = count_parameters(model)
    size_bytes = trainable * 4
    size_mb = size_bytes / (1024 ** 2)
    return size_mb


# ==========================
#  Dane: CIFAR-10
# ==========================
def get_cifar10_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, data_dir=DATA_DIR):
    # Statystyki CIFAR-10 (standardowe)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader


# ==========================
#  Model: MobileNetV2 na CIFAR-10
# ==========================
def get_mobilenet_v2_cifar10(num_classes=10, pretrained=USE_PRETRAINED):
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

    if pretrained:
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        model = mobilenet_v2(weights=weights)
    else:
        model = mobilenet_v2(weights=None)

    # Podmiana klasyfikatora na 10 klas
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


# ==========================
#  Trening / ewaluacja
# ==========================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Eval", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(
    model,
    train_loader,
    test_loader,
    epochs,
    lr,
    device,
    description="base"
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for epoch in range(1, epochs + 1):
        print(f"\n[{description}] Epoch {epoch}/{epochs}")
        start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc

        elapsed = time.time() - start
        print(
            f"[{description}] "
            f"Train loss: {train_loss:.4f}, acc: {train_acc*100:.2f}% | "
            f"Test loss: {test_loss:.4f}, acc: {test_acc*100:.2f}% "
            f"(time: {elapsed:.1f}s)"
        )

    return model, history, best_acc


# ==========================
#  Pruning
# ==========================
def apply_global_pruning(model: nn.Module, amount: float = 0.5):
    """
    Globalny pruning L1 wag Conv2d i Linear.
    amount = 0.5 oznacza, że usuwamy 50% wag o najmniejszej wartości bezwzględnej.
    """
    parameters_to_prune = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, "weight"))

    print(f"Liczba warstw poddanych pruningowi: {len(parameters_to_prune)}")

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    # Usunięcie reparametryzacji (przepisanie wag z powrotem do "weight")
    for module, _ in parameters_to_prune:
        prune.remove(module, "weight")

    return model


# ==========================
#  Główny pipeline
# ==========================
def main():
    set_seed(SEED)
    device = get_device()
    print(f"Urządzenie: {device}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Dane
    print("Przygotowanie loaderów CIFAR-10...")
    train_loader, test_loader = get_cifar10_loaders()

    # 2. Model bazowy
    print("Tworzenie modelu MobileNetV2...")
    base_model = get_mobilenet_v2_cifar10()
    base_model.to(device)

    # Info o parametrach
    total_params, trainable_params = count_parameters(base_model)
    size_mb = estimate_model_size_mb(base_model)
    print(f"Model bazowy - liczba parametrów: {trainable_params} ({total_params} łącznie)")
    print(f"Szacowany rozmiar modelu: {size_mb:.2f} MB")

    # 3. Trening modelu bazowego
    print("\n== Trening modelu bazowego ==")
    base_model, base_history, base_best_acc = train_model(
        base_model,
        train_loader,
        test_loader,
        epochs=BASE_EPOCHS,
        lr=BASE_LR,
        device=device,
        description="base",
    )

    torch.save(base_model.state_dict(), os.path.join(SAVE_DIR, "mobilenetv2_cifar10_base.pt"))
    print(f"Zapisano model bazowy. Najlepsze test accuracy: {base_best_acc*100:.2f}%")

    # 4. Pruning (na wytrenowanym modelu)
    print("\n== Pruning modelu ==")
    pruned_model = deepcopy(base_model)  # opcjonalnie kopiujemy, by mieć bazę osobno
    pruned_model = apply_global_pruning(pruned_model, amount=PRUNE_AMOUNT)
    pruned_model.to(device)

    # Info po pruning
    total_params_pruned, trainable_params_pruned = count_parameters(pruned_model)
    size_mb_pruned = estimate_model_size_mb(pruned_model)
    print(f"Model po pruning - liczba parametrów: {trainable_params_pruned} ({total_params_pruned} łącznie)")
    print(f"Szacowany rozmiar po pruning: {size_mb_pruned:.2f} MB")

    # 5. Fine-tuning po pruning
    print("\n== Fine-tuning po pruning ==")
    pruned_model, pruned_history, pruned_best_acc = train_model(
        pruned_model,
        train_loader,
        test_loader,
        epochs=FINETUNE_EPOCHS,
        lr=FINETUNE_LR,
        device=device,
        description="pruned_finetune",
    )

    torch.save(pruned_model.state_dict(), os.path.join(SAVE_DIR, "mobilenetv2_cifar10_pruned.pt"))
    print(f"Zapisano model po pruning. Najlepsze test accuracy: {pruned_best_acc*100:.2f}%")

    # 6. Podsumowanie
    print("\n==================== PODSUMOWANIE ====================")
    print(f"Model bazowy:")
    print(f"  Accuracy (test): {base_best_acc*100:.2f}%")
    print(f"  Parametry: {trainable_params} (~{size_mb:.2f} MB)")

    print(f"\nModel po pruning + fine-tuning (amount={PRUNE_AMOUNT}):")
    print(f"  Accuracy (test): {pruned_best_acc*100:.2f}%")
    print(f"  Parametry: {trainable_params_pruned} (~{size_mb_pruned:.2f} MB)")

    redukcja_param = 100.0 * (1.0 - trainable_params_pruned / trainable_params)
    print(f"\nRedukcja liczby parametrów: {redukcja_param:.2f}%")
    print("=======================================================")


if __name__ == "__main__":
    main()
