import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import TensorDataset

df = pd.read_csv("worldcities.csv")
df = df.replace(r'^\s*$', np.nan, regex=True)
df = df.dropna()

le = LabelEncoder()
le.fit(df["capital"])
Y = le.transform(df["capital"])
X = df[["lat", "lng", "population"]]


train_X, val_X, train_Y, val_Y = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=15,
)
train_X, val_X = train_X.to_numpy(), val_X.to_numpy()

train_X, val_X, train_Y, val_Y = torch.tensor(train_X, dtype=torch.float32), torch.tensor(val_X, dtype=torch.float32), torch.tensor(train_Y, dtype=torch.long), torch.tensor(val_Y, dtype=torch.long)
print(train_X.shape, train_Y.shape, val_X.shape, val_Y.shape)
train_ds = TensorDataset(train_X, train_Y)
val_ds = TensorDataset(val_X, val_Y)
from torch.utils.data import DataLoader

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 3)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def eval(model, city_lat, city_lot, city_population):
    X = torch.tensor([[city_lat, city_lot, city_population]]).to(device)
    pred = model(X)
    pred = pred.argmax(1)
    print(le.inverse_transform(pred)[0])

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(val_loader, model, loss_fn)
print("Done!")

eval(model, 55.8657, 9.8735, 63162)