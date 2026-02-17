import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

root = "./Data_10"
batch_size = 10

transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

train_set = datasets.CIFAR10(root=root, train=True, transform=transformations, download=True)

test_set = datasets.CIFAR10(root=root, train=False, transform=transformations, download=True)

train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

print(f"Количество тренировочных изображений: {len(train_set)}")
print(f"Количество тестовых изображений: {len(test_set)}")
print(f"Размер батча: {batch_size}")
print(f"Количество батчей в тренировочных данных: {len(train_data_loader)}")
print(f"Количество батчей в тестовых данных: {len(test_data_loader)}")

for images, labels in train_data_loader:
    print(f"Размерность батча изображений: {images.shape}")
    print(f"Размерность батча меток: {labels.shape}")
    break