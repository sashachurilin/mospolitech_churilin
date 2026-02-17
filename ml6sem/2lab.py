import torch
import torch.nn as nn

#  1. Создание нейросети с тремя скрытыми слоями 
class ThreeHiddenLayerNet(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, hidden3_size, output_size):
        super().__init__()
        
        #  Слой 1 
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.activation1 = nn.ReLU()

        #  Слой 2 
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.activation2 = nn.Tanh()

        #  Слой 3 
        self.fc3 = nn.Linear(hidden2_size, hidden3_size)
        self.activation3 = nn.LeakyReLU()

        self.fc4 = nn.Linear(hidden3_size, output_size)
        
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation1(out)
        out = self.fc2(out)
        out = self.activation2(out)
        out = self.fc3(out)
        out = self.activation3(out)
        out = self.fc4(out)
        out = self.output_activation(out)
        return out

input_size = 5
hidden1_size = 16
hidden2_size = 12
hidden3_size = 8
output_size = 1
learning_rate = 0.01
num_epochs = 200

model = ThreeHiddenLayerNet(input_size, hidden1_size, hidden2_size, hidden3_size, output_size)
print(model)
print("-" * 30)

criterion = nn.BCELoss()

#  3. Попробуйте поменять оптимизатор
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

x_train = torch.rand(100, input_size)
y_train = torch.randint(0, 2, (100, 1)).float()


for epoch in range(num_epochs):
    optimizer.zero_grad()
    out = model(x_train)
    loss = criterion(out, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Эпоха [{epoch + 1}/{num_epochs}], Потери: {loss.item():.4f}')

print("-" * 30)

# Какой оптимизатор лучше?
# 1. Adam: быстро сходится к хорошему результату, потери падают стабильно и быстро
# 2. SGD: сходится медленнее, потери могут 'прыгать'
# Adam лучше
print("-" * 30)

x_test = torch.rand(10, input_size)
with torch.no_grad():
    predictions = model(x_test)
