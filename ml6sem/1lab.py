import torch
import torch.nn as nn
# import os

#  1. Проверка установки PyTorch 
print(f"PyTorch версия: {torch.__version__}")
print("-" * 30)

#  2. Создание и вывод двух тензоров 3x3 
tensor1 = torch.rand(3, 3)
tensor2 = torch.rand(3, 3)
print("Тензор 1:\n", tensor1)
print("\nТензор 2:\n", tensor2)
print("-" * 30)

#  3. Сложение тензоров 
sum_tensor = tensor1 + tensor2
print("Сумма тензоров:\n", sum_tensor)
print("-" * 30)

#  4. Поэлементное умножение 
mul_tensor = tensor1 * tensor2
print("Поэлементное умножение:\n", mul_tensor)
print("-" * 30)

#  5. Транспонирование второго тензора 
transposed_tensor2 = tensor2.T
print("Транспонированный тензор 2:\n", transposed_tensor2)
print("-" * 30)

#  6. Среднее значение в каждом тензоре 
mean_tensor1 = torch.mean(tensor1)
mean_tensor2 = torch.mean(tensor2)
print(f"Среднее значение в тензоре 1: {mean_tensor1:.4f}")
print(f"Среднее значение в тензоре 2: {mean_tensor2:.4f}")
print("-" * 30)

#  7. Максимальное значение в каждом тензоре 
max_tensor1 = torch.max(tensor1)
max_tensor2 = torch.max(tensor2)
print(f"Максимальное значение в тензоре 1: {max_tensor1:.4f}")
print(f"Максимальное значение в тензоре 2: {max_tensor2:.4f}")
print("-" * 30)

#  8. Создание и обучение нейросети для умножения 
class MultiplicationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

training_samples = 100
num_epochs = 500
learning_rate = 0.01

X_train = torch.rand(training_samples, 2) * 10
y_train = (X_train[:, 0] * X_train[:, 1]).unsqueeze(1)

model = MultiplicationNet()
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# for epoch in range(num_epochs):
#     optimizer.zero_grad()
#     out = model(X_train)
#     error = loss_func(out, y_train)
#     error.backward()
#     optimizer.step()

#     if (epoch + 1) % 100 == 0:
#         print(f'Эпоха [{epoch + 1}/{num_epochs}], Потеря: {error.item():.4f}')
print("-" * 30)

#  9. 10 проверочных пар чисел 
X_test = torch.rand(10, 2) * 10
y_test = (X_test[:, 0] * X_test[:, 1])

model.eval()
with torch.no_grad():
    predictions = model(X_test).squeeze()

print("Вход (A, B)   | Ожидаемый (A*B) | Результат сети | Разница")
for i in range(10):
    diff = abs(y_test[i] - predictions[i])
    print(f"({X_test[i, 0]:.2f}, {X_test[i, 1]:.2f}) | {y_test[i]:<17.4f} | {predictions[i]:<16.4f} | {diff:.4f}")
print("-" * 30)

#  10. Как изменился результат? 
# 10: модель, будет неточной, 'потеря' останется большой
# 1000: модель станет значительно точнее, 'потеря' будет меньше

#  11. Сохранение модели
model_path = 'Neuro.pth'
torch.save(model.state_dict(), model_path)
print("-" * 30)

#  12. Загрузка модели из файла
loaded_model = MultiplicationNet()
loaded_model.load_state_dict(torch.load(model_path))
print("-" * 30)

#  13. Убедитесь, что модель по-прежнему работает правильно 
loaded_model.eval()
with torch.no_grad():
    loaded_predictions = loaded_model(X_test).squeeze()

print("Вход (A, B)   | Ожидаемый (A*B) | Результат сети | Разница")
for i in range(10):
    diff = abs(y_test[i] - loaded_predictions[i])
    print(f"({X_test[i, 0]:.2f}, {X_test[i, 1]:.2f}) | {y_test[i]:<17.4f} | {loaded_predictions[i]:<16.4f} | {diff:.4f}")

# if os.path.exists(model_path):
#     os.remove(model_path)