import argparse
import errno
import torch
import os 

from dataset import Dataset
from model import CNN

from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_CHECKPOINTS = CURRENT_DIR_PATH + '/models/models_checkpoints/'

def make_args_parser():
    # create an ArgumentParser object
    parser = argparse.ArgumentParser(description='CNN Based Wideband Spectrum Occupancy Status Identification for Cognitive Radios')
    # fill parser with information about program arguments
    parser.add_argument('-d', '--data_folder', type=str, default='data',
                    	help='Define the name of the folder where datasets are stored.')
    parser.add_argument('-e', '--epochs', type=int, default=1000,
                    	help='Define the number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=16,
                        help='Define the batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help='Define the learning rate')

    # return an ArgumentParser object
    return parser.parse_args()

def print_args(args):
    print("Running with the following configuration")
    # get the __dict__ attribute of args using vars() function
    args_map = vars(args)
    for key in args_map:
        print('\t', key, '-->', args_map[key])
    # add one more empty line for better output
    print()


def main():
	# Parse and print arguments
	args = make_args_parser()
	print_args(args)
	# Check device available
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print("Running on: {}".format(device))
	# Create directory to save model's checkpoints
	try:
	    model_root = MODEL_CHECKPOINTS
	    os.makedirs(model_root)
	except OSError as e:
	    if e.errno == errno.EEXIST:
	        pass
	    else:
	        raise
	# Load, split to train/val and normalize data 
	transform = transforms.Compose([
		transforms.ToTensor(),
     	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     	])
	data = np.load(os.path.join(CURRENT_DIR_PATH, args.data_folder, 'train/data.npy'))
	labels = np.load(os.path.join(CURRENT_DIR_PATH, args.data_folder, 'train/labels.npy'))
	X_train, X_val, y_train, y_val = train_test_split(data, labels test_size=0.20, random_state=42)
	train_dataset = Dataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long(), transform=transform)
	val_dataset = Dataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long(), transform=transform)
	trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
	valloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

	# Define NN
	net = CNN()

	# Define a Loss function and optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

	# Training loop
	for epoch in range(args.epochs):
	    running_loss = 0.0
	    for i, data in enumerate(trainloader, 0):
	        # get the inputs; data is a list of [inputs, labels]
	        inputs, labels = data

	        # zero the parameter gradients
	        optimizer.zero_grad()

	        # forward + backward + optimize
	        outputs = net(inputs)
	        loss = criterion(outputs, labels)
	        loss.backward()
	        optimizer.step()

	        # print statistics
	        running_loss += loss.item()
	        if i % 2000 == 1999:    # print every 2000 mini-batches
	            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
	            running_loss = 0.0




if __name__ == "__main__":
	main()
