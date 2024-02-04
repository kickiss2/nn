import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import random
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true')
parser.add_argument('--predict')
args = parser.parse_args()

model_name = 'model.pth'
learning_rate = 0.001
input_size = 3
hidden_size = 32
batch_size = 1
num_epochs = 50

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Define a custom PyTorch Dataset
class LocalMinimaDataset(Dataset):
    def __init__(self, candles, labels):
        self.candles = candles
        self.labels = labels

    def __len__(self):
        return len(self.candles)

    def __getitem__(self, idx):
        return self.candles[idx], self.labels[idx]

# Define the neural network model
class LocalMinimaPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(LocalMinimaPredictor, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

model = LocalMinimaPredictor(input_size, hidden_size)

############################################# training ###############################################

if args.train:
    model.train()
    
    data_x = np.array([
                [1, 2, 3],
                [2, 3, 4],
                [3, 4, 5],
                [4, 5, 6],
                [5, 6, 7],
                [6, 7, 8],
                [7, 8, 9]
            ])

    data_y = np.array([
                [1], 
                [0], 
                [1], 
                [0], 
                [1], 
                [0], 
                [1]
            ])

    data_x = torch.FloatTensor(data_x)
    data_y = torch.FloatTensor(data_y)

    print(data_x)
    print(data_y)

    dataset = LocalMinimaDataset(data_x, data_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    # Train the model
    num_epochs = 50
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate the loss
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Update the model parameters
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
        torch.save(model.state_dict(), model_name)
    exit()

############################################# predicting ###############################################

if args.predict:
    model.load_state_dict(torch.load(model_name))

    data = np.array([args.predict.split(",")])
    data = data.astype(np.float32)

    input = torch.FloatTensor(data)

    with torch.no_grad():
        out = model(input)
        out = out.cpu().detach().numpy()

        print(input)
        print(out)
    exit()

print("Please specify either --train or --predict")
print("Example1: python simplified.py --train")
print("Example2: python simplified.py --predict=1,2,3")


# root@1930e186ca79:/app/test# python simplified.py --predict=1,2,3
# tensor([[1., 2., 3.]])
# [[0.5537224]]
# root@1930e186ca79:/app/test# python simplified.py --predict=2,3,4
# tensor([[2., 3., 4.]])
# [[0.5602484]]
#
# :(((((((

