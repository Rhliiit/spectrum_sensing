import random 
import os 

import numpy as np

from datetime import datetime
from joblib import Parallel, delayed
from numpy import sum, isrealobj, sqrt
from numpy.random import standard_normal
from tqdm import tqdm
from pathlib import Path

RUN_TYPE = 'parallel'

CWSS_area = (600, 600)
PU_coords = np.array((280, 320))
PU_active_prob = 0.5
SNR = 0.5   
Q = 16
w = 5e+6
N_p = 6
K = 12 
M = 36
tau = 1e-4
alpha = 4
delta_0 = 1.0
P_PU = 2.0
train_samples = 10000
test_samples = 5000

def generate_SUs_coords():
	"""
	In simulations, we consider that the SUs participating in CWSS are randomly deployed 
	in an 600m X 600m area. Without loss of generality, we choose to have integer coordinates 
	for each SU. 
	We get the coordinates of the m-th SU by indexing SUs_coords list by m-1: SUs_coords[m-1]
	"""
	SUs_coords = []
	for _ in range(M):
		coords = (random.randint(0,  CWSS_area[0]), random.randint(0,  CWSS_area[0]))
		SUs_coords.append(np.array(coords))
	return SUs_coords

SUs_coords = generate_SUs_coords()

def compute_channel_gain(PU_coords, SU_coords):
	def compute_power_loss(value):
		return value ** (-alpha)

	def compute_euclidean_distance(a, b):
		return np.linalg.norm(a-b)

	return np.sqrt(compute_power_loss(compute_euclidean_distance(PU_coords, SU_coords)))

def compute_transmit_symbol(num_bits):
	return sum(np.random.randint(2, size=(num_bits,)))

def awgn(s, SNRdB, L=1):
	"""
	AWGN channel
	Add AWGN noise to input signal. The function adds AWGN noise vector to signal 's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
	returns the noise vector 'n' that is added to the signal 's' and the power spectral density N0 of noise added
	Parameters:
	    s : input/transmitted signal vector
	    SNRdB : desired signal to noise ratio (expressed in dB) for the received signal
	    L : oversampling factor (applicable for waveform simulation) default L = 1.
	Returns:
	    r : received signal vector (r=s+n)
	"""
	gamma = 10**(SNRdB/10) 										# SNR to linear scale
	P=L*sum(abs(s)**2)											# Actual power in the vector
	N0=P/gamma 													# Find the noise spectral density
	if isrealobj(s):											# check if input is real/complex object type
	    n = sqrt(N0/2)*standard_normal(s.shape) 				# computed noise
	else:
	    n = sqrt(N0/2)*(standard_normal(s.shape)+1j*standard_normal(s.shape))
	r = s + n 													# received signal
	return r

def generate_energy_matrix(data, labels, t):
	# X is the energy matrix of shape (Q, M) initialized to zeros 
	X = np.zeros((Q, M))
	# Creating a balanced dataset by selecting the label using modulo
	L = t % K
	# Compute energy matrix
	for m in range(M):
		for q in range(Q):
			# Check if q is within the occupied subband of PU, while PU is active
			if random.uniform(0, 1) >= PU_active_prob:	
				if L != K-1 and q >= L and q <= L+N_p-1:
					PU_transmit_power = np.sqrt(P_PU)
				else:
					PU_transmit_power = 0
			else: 
				PU_transmit_power = 0
			# 2 * w * tau denotes the number of samples in a sensing interval,
			total = 0.0
			for _ in range(int(2*w*tau)):
				y = PU_transmit_power * compute_channel_gain(PU_coords, SUs_coords[m]) * compute_transmit_symbol(int(2*w*tau))
				y += awgn(np.array(y), SNR)
				total += y**2
			X[q, m] = total
	data[t, :, :] = X
	labels[t] = L

def main(num_samples, export_dirpath=''):
	# Initialize matrices for data and labels
	global data, labels
	data = np.zeros((num_samples, Q, M))
	labels = np.zeros((num_samples,))
	
	if RUN_TYPE == 'sequentally':
		for t in tqdm(range(num_samples), position=0, leave=True):
			generate_energy_matrix(data, labels, t)
	elif RUN_TYPE == 'parallel':
		Parallel(n_jobs=-1, backend="threading")(delayed(generate_energy_matrix)(data, labels, t) for t in tqdm(range(num_samples), position=0, leave=True))
	
	else:
		print("Unknown run type: ", RUN_TYPE)
		print("Exiting...")
		exit()

	# Save data/labels as numpy arrays
	path = Path(export_dirpath)
	path.mkdir(parents=True, exist_ok=True)

	with open(os.path.join(export_dirpath, 'data.npy'), 'wb') as f:
		np.save(f, data)
	
	with open(os.path.join(export_dirpath, 'labels.npy'), 'wb') as f:
		np.save(f, labels)


if __name__ == "__main__":
	print("[{}]: Generating training set...".format(datetime.now().strftime("%H:%M:%S")))
	main(num_samples=train_samples, export_dirpath='data/train')
	print("[{}]: Generating training set completed.".format(datetime.now().strftime("%H:%M:%S")))
	print("[{}]: Generating testing set...".format(datetime.now().strftime("%H:%M:%S")))
	main(num_samples=test_samples, export_dirpath='data/test')
	print("[{}]: Generating testing set completed.".format(datetime.now().strftime("%H:%M:%S")))
