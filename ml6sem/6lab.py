import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
import time

# # for google colab
# df = pd.read_csv('/opt/for 6 lab.csv')

# lecture method
df = pd.read_csv('for 6 lab.csv')
df.dropna(inplace=True)

all_reviews = df['review'].tolist()
encoded_labels = np.array(df['label'].tolist())

# # manually run everywere (small data)
# 
# positive_reviews = [
#     'this movie was absolutely fantastic a masterpiece of modern cinema',
#     'i love this product it works perfectly and exceeded all my expectations',
#     'the service was excellent the staff was friendly and very helpful',
#     'a truly wonderful experience from start to finish highly recommend',
#     'great quality and value for money i would definitely buy it again',
#     'the book is a compelling read i could not put it down',
#     'amazing performance by the actors the plot was gripping',
#     'the food was delicious and the atmosphere was very cozy',
#     'this app is incredibly useful and easy to navigate five stars',
#     'best purchase i have made this year solid and reliable'
# ]

# negative_reviews = [
#     'a complete waste of time and money the plot was predictable and boring',
#     'the product broke after just one week of use terrible quality',
#     'i am very disappointed with the service the staff was rude',
#     'a truly awful experience i would not recommend this place to anyone',
#     'terrible value for money it feels cheap and poorly made',
#     'the book was a drag to get through the writing is simply bad',
#     'i hated this film it was two hours of my life i will never get back',
#     'the food was cold and tasteless the restaurant was dirty',
#     'this app crashes constantly and is full of bugs completely unusable',
#     'worst purchase ever it did not work as advertised do not buy'
# ]

# all_reviews = positive_reviews + negative_reviews

# labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)
# encoded_labels = np.array(labels)

all_text = ' '.join(all_reviews)
words = all_text.split()
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
word2int = {word: i + 1 for i, word in enumerate(vocab)}

reviews_int = []
for review in all_reviews:
    reviews_int.append([word2int.get(word, 0) for word in review.split()])

seq_len = 256
features = np.zeros((len(reviews_int), seq_len), dtype=int)
for i, review in enumerate(reviews_int):
    review_len = len(review)
    if review_len <= seq_len:
        features[i, -review_len:] = np.array(review)
    else:
        features[i, :] = np.array(review)[:seq_len]

split_frac = 0.8
split_idx = int(len(features) * split_frac)
train_x, remaining_x = features[:split_idx], features[split_idx:]
train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

test_idx = int(len(remaining_x) * 0.5)
val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

batch_size = 50
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, lstm_layers, drop_prob=0.5):
        super(SentimentLSTM, self).__init__()
        self.output_size = output_size
        self.lstm_layers = lstm_layers
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, lstm_layers,
                            dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        out = self.dropout(lstm_out)
        out = self.fc(out)
        sig_out = self.sig(out)
        
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]
        
        return sig_out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.lstm_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.lstm_layers, batch_size, self.hidden_dim).zero_())
        return hidden

def run_training(lstm_layers_count):
    start_time = time.time()

    vocab_size = len(word2int) + 1
    output_size = 1
    embedding_dim = 400
    hidden_dim = 256

    model = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, lstm_layers_count)
    
    lr = 0.001
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    epochs = 4
    clip = 5
    best_valid_loss = float('inf')
    model_save_path = f'lern_lstm_{lstm_layers_count}.pth'

    for i in range(epochs):
        h = model.init_hidden(batch_size)
        model.train()
        train_accuracy = 0.0

        for inputs, labels in train_loader:
            h = tuple([each.data for each in h])
            model.zero_grad()
            output, h = model(inputs, h)
            loss = loss_function(output.squeeze(), labels.float())
            loss.backward()
            
            pred = torch.round(output.squeeze())
            correct_tensor = pred.eq(labels.float().view_as(pred))
            train_accuracy += np.sum(np.squeeze(correct_tensor.numpy()))
            
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        
        val_h = model.init_hidden(batch_size)
        val_losses = []
        test_accuracy = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                val_h = tuple([each.data for each in val_h])
                output, val_h = model(inputs, val_h)
                val_loss = loss_function(output.squeeze(), labels.float())
                val_losses.append(val_loss.item())

                pred = torch.round(output.squeeze())
                correct_tensor = pred.eq(labels.float().view_as(pred))
                test_accuracy += np.sum(np.squeeze(correct_tensor.numpy()))

        epoch_val_loss = np.mean(val_losses)
        
        print(f"Эпоха: {i+1}/{epochs}...",
              f"Точность train: {train_accuracy / len(train_loader.dataset) * 100:.2f}%...",
              f"Точность test: {test_accuracy / len(valid_loader.dataset) * 100:.2f}%")

        if epoch_val_loss < best_valid_loss:
            best_valid_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
    
    end_time = time.time()
    total_time = end_time - start_time
    return total_time

time_for_2_layers = run_training(lstm_layers_count=2)
time_for_4_layers = run_training(lstm_layers_count=4)

print(f"Время обучения (2 слоя): {time_for_2_layers:.2f} секунд.")
print(f"Время обучения (4 слоя): {time_for_4_layers:.2f} секунд.")