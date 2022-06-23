import torch
import torch.utils.data as data

class Dataset(data.Dataset):
	def __init__(self, X_data, y_data, transform=None):
		self.transform = transform
		if self.transform:
			self.X_data = transform(X_data)
		else:
			self.X_data = X_data
		self.y_data = y_data

	def __getitem__(self, index):
		return self.X_data[index], self.y_data[index]
	
	def __len__ (self):
		return len(self.X_data)