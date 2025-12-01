import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from torchvision.models import mobilenet_v2


# =========================================
#  Ładowanie CIFAR-10
# =========================================
def get_cifar10_testloader(batch_size=128):
    # CIFAR-10 jest już pobrany ręcznie
    # Struktura:
    # ESL_Tiny-Image-Classifier/
    #   data/cifar-10-batches-py/
    
    import os

    data_root = os.path.join(os.path.dirname(__file__), "..", "data")

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    test_set = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=False,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return test_loader


# =========================================
#  Wczytanie modelu MobileNetV2 dla CIFAR-10
# =========================================
def load_mobilenet_model(weights_path, device):
    model = mobilenet_v2(weights=None)

    
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 10)

    # Wczytywanie wag wytrenowanych przez nas
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


# =========================================
#  Liczenie accuracy
# =========================================
def evaluate(model, test_loader, device):
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)

            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    return acc


# =========================================
#  Policzenie liczby parametrów
# =========================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# =========================================
#  Policzenie liczby NIEZEROWYCH wag (sparsity)
# =========================================
def count_nonzero_params(model):
    nonzero = 0
    total = 0
    for p in model.parameters():
        if p is not None:
            nz = torch.count_nonzero(p).item()
            nonzero += nz
            total += p.numel()
    sparsity = 100.0 * (1.0 - nonzero / total)
    return nonzero, total, sparsity



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie: {device}")

    test_loader = get_cifar10_testloader()

    print("\nŁadowanie modeli...")

    model_base = load_mobilenet_model("mobilenetv2_cifar10_base.pt", device)
    model_pruned = load_mobilenet_model("mobilenetv2_cifar10_pruned.pt", device)

    print("\n== Ewaluacja ==")

    base_acc = evaluate(model_base, test_loader, device)
    pruned_acc = evaluate(model_pruned, test_loader, device)

    print("\nWyniki:")
    print(f"Base model accuracy:  {base_acc:.2f}%")
    print(f"Pruned model accuracy: {pruned_acc:.2f}%")

    print("\nParametry:")
    print(f"Base model params:   {count_parameters(model_base)}")
    print(f"Pruned model params: {count_parameters(model_pruned)}")

    # Policzenie sparsity (procent wag wyzerowanych)
    nz_base, total_base, sparsity_base = count_nonzero_params(model_base)
    nz_pruned, total_pruned, sparsity_pruned = count_nonzero_params(model_pruned)

    print("\nSparsity:")
    print(f"Base model sparsity:   {sparsity_base:.2f}% zer")
    print(f"Pruned model sparsity: {sparsity_pruned:.2f}% zer")
    print(f"Non-zero params base:   {nz_base}")
    print(f"Non-zero params pruned: {nz_pruned}")



if __name__ == "__main__":
    main()
