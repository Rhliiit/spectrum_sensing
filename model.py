import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc4 = nn.Linear(4608, 400)
        self.fc5 = nn.Linear(400, 12)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


if __name__ == "__main__":
	B = 64	# batch size
	Q = 16	# number of subbands
	M = 36	# number of SUs

	# Generate random input data for testing
	data = torch.rand((B, 1, Q, M))
	print("------------------------------------- Input Data -------------------------------------")
	print(data)
	print(data.shape)
	# Build CNN
	net = CNN()
	print("---------------------------------- CNN architecture ----------------------------------")
	print(net)
	# Execture a forward pass
	print("------------------------------------ Forward Pass ------------------------------------")
	out = net(data)
	assert out.shape == (B, 12)
	print(out.shape)