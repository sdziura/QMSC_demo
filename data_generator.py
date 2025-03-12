"""
This script generates a Hidden Markov Model data with two observables.
Data will consist a set of fixed length chains of observed variables, 
generated from two different models, with the labels informing which model 
was used to generate this chain. 

Generated dataset simulates smartwatch sensors.Hidden states are the type 
of activity the user is taking (walking, running cycling or driving).
Observables are acceleration (m/s^2) and gyroscope (rad/s) sensors readings.

High entropy of transitions of hidden states can mean that sensors readings are noisy 
or that smartwatch is not used properly. Such readings should be classified
and rejected.

"""
import numpy as np
from hmmlearn import hmm
import h5py

# Set random seed for reproducibility
np.random.seed(42)

# Define the number of hidden states and observables
n_hidden = 4   # Hidden states: e.g., "Walking", "Running", "Cycling", "Driving"
n_observables = 2  # Two observed features: acceleration, gyroscope

# Set the initial state probabilities
startprob = np.array([0.25, 0.25, 0.25, 0.25])  

# Transition matrices
low_entropy_transitions = np.array([
    [0.9, 0.05, 0.025, 0.025],  # Walking
    [0.05, 0.9, 0.025, 0.025],  # Running
    [0.05, 0.025, 0.9, 0.025],  # Cycling
    [0.05, 0.025, 0.025, 0.9]   # Driving
])

high_entropy_transitions = np.full((4, 4), 0.25)

# Mean acceleration and gyroscope values
means = np.array([
    [1.0, 0.5],  # Walking
    [3.0, 2.0],  # Running
    [2.0, 3.5],  # Cycling
    [4.0, 1.5]   # Driving 
])

# Variances (uncertainty)
covars = np.array([
    [0.2, 0.1],  # Walking
    [0.5, 0.3],  # Running
    [0.4, 0.6],  # Cycling
    [0.6, 0.2]   # Driving 
])

# Initialize the HMM with low entropy
model_low_entropy = hmm.GaussianHMM(n_components=n_hidden, covariance_type="diag")
model_low_entropy.startprob_ = startprob
model_low_entropy.transmat_ = low_entropy_transitions
model_low_entropy.means_ = means
model_low_entropy.covars_ = covars

# Initialize the HMM with high entropy
model_high_entropy = hmm.GaussianHMM(n_components=n_hidden, covariance_type="diag")
model_high_entropy.startprob_ = startprob
model_high_entropy.transmat_ = high_entropy_transitions
model_high_entropy.means_ = means
model_high_entropy.covars_ = covars

# Calculate entropy
entropy = -np.nansum(low_entropy_transitions[0] * np.log2(low_entropy_transitions[0]))
entropy_h = -np.nansum(high_entropy_transitions[0] * np.log2(high_entropy_transitions[0]))
print("Entropy of good data:", entropy)
print("Entropy of bad data:", entropy_h)


# Generate sequences
num_chains = 1000
chain_length = 10

hidden_sequences_low = []
observed_sequences_low= []

hidden_sequences_high = []
observed_sequences_high = []

for _ in range(num_chains):
    obs_seq_low, _ = model_low_entropy.sample(chain_length)
    #hidden_sequences_low.append(hidden_seq_low.flatten())
    observed_sequences_low.append(obs_seq_low)

    obs_seq_high, _= model_high_entropy.sample(chain_length)
    #hidden_sequences_high.append(hidden_seq_high.flatten())
    observed_sequences_high.append(obs_seq_high)

# Convert lists to NumPy arrays
#hidden_sequences_low = np.array(hidden_sequences_low)
observed_sequences_low = np.array(observed_sequences_low)
y_low = np.zeros(len(observed_sequences_low))
#hidden_sequences_high = np.array(hidden_sequences_high)
observed_sequences_high = np.array(observed_sequences_high)
y_high = np.ones(len(observed_sequences_low))

# concatenate 2 categories and shuffle
y = np.concatenate([y_low, y_high], axis=0)
observations = np.concatenate([observed_sequences_low, observed_sequences_high], axis=0)
indices = np.arange(len(y))
np.random.shuffle(indices)
observations = observations[indices]
labels = y[indices]

# Save to HDF5
with h5py.File("hmm_gaussian_chains.h5", "w") as f:
    f.create_dataset("label", data=y)
    f.create_dataset("observed", data=observations)

print("Saved sequences to hmm_gaussian_chains.h5")

