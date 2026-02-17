import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# 1. ядер 3x3
class CnnModel_v1(nn.Module):
    def __init__(self, num_classes=10):
        super(CnnModel_v1, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(24)
        
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)

        self.fc = nn.Linear(24 * 16 * 16, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.pool(out)
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
        
# 1. Скорость: 3x3 обычно работает БЫСТРЕЕ, т.к. меньше параметров = меньше вычислений
# 2. Точность: увеличилась

# 2. Новая архитектура
class CnnModel_v2(nn.Module):
    def __init__(self, num_classes=10):
        super(CnnModel_v2, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.fc = nn.Linear(32 * 16 * 16, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.pool1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        out = out.view(out.size(0), -1)
        
        out = self.fc(out)
        return out

# 1. Скорость: быстрее, т.к. меньше слоев и параметров
# 2. Точность: ниже

# модель для 3, 4, 5

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_set = torchvision.datasets.CIFAR10(root='./Data_10', train=False, download=False, transform=transform)

# 3. Вытащить 20 картинок
test_loader = torch.utils.data.DataLoader(test_set, batch_size=20, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = CnnModel_v2(num_classes=10)
model.eval()

with torch.no_grad():
    images, labels = next(iter(test_loader))
    
    outputs = model(images)
    
    _, predicted = torch.max(outputs, 1)

    
    # 4. Вывод результатов
    print("{:<15} {:<15}".format("Прогноз сети", "Правильный ответ"))
    print("-" * 30)
    
    correct_count = 0
    for i in range(20):
        predicted_class = classes[predicted[i]]
        true_class = classes[labels[i]]
        print(f"{predicted_class:<15} {true_class:<15}")
        if predicted_class == true_class:
            correct_count += 1

    # 5. Расчет точности 
    accuracy = (correct_count / 20) * 100
    
    print("-" * 40)
    print(f"Угадано правильно: {correct_count} из 20.")
    print(f"Точность на этой выборке: {accuracy:.2f}%")