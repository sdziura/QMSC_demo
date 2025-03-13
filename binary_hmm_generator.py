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


class BinaryHMMGenerator:

    def __init__(
        self,
        n_hidden=4,
        n_observables=2,
        startprob=np.array([0.25, 0.25, 0.25, 0.25]),
        low_entropy_transitions=np.array(
            [
                [0.9, 0.05, 0.025, 0.025],  # Walking
                [0.05, 0.9, 0.025, 0.025],  # Running
                [0.05, 0.025, 0.9, 0.025],  # Cycling
                [0.05, 0.025, 0.025, 0.9],  # Driving
            ]
        ),
        high_entropy_transitions=np.full((4, 4), 0.25),
        means=np.array(
            [
                [1.0, 0.5],  # Walking
                [3.0, 2.0],  # Running
                [2.0, 3.5],  # Cycling
                [4.0, 1.5],  # Driving
            ]
        ),
        covars=np.array(
            [
                [0.2, 0.1],  # Walking
                [0.5, 0.3],  # Running
                [0.4, 0.6],  # Cycling
                [0.6, 0.2],  # Driving
            ]
        ),
        seed=42,
    ):
        # Set random seed for reproducibility
        np.random.seed(seed)

        # Define the number of hidden states and observables
        self.n_hidden = (
            n_hidden  # Hidden states: e.g., "Walking", "Running", "Cycling", "Driving"
        )
        self.n_observables = (
            n_observables  # Two observed features: acceleration, gyroscope
        )

        # Set the initial state probabilities
        self.startprob = startprob

        # Transition matrices
        self.low_entropy_transitions = low_entropy_transitions
        self.high_entropy_transitions = high_entropy_transitions

        # Mean acceleration and gyroscope values for each state
        self.means = means

        # Variances (uncertainty)
        self.covars = covars

    def gen_binary_hmm(self, samples=2000, chain_length=10):
        # Initialize the HMM with low entropy
        model_low_entropy = hmm.GaussianHMM(
            n_components=self.n_hidden, covariance_type="diag"
        )
        model_low_entropy.startprob_ = self.startprob
        model_low_entropy.transmat_ = self.low_entropy_transitions
        model_low_entropy.means_ = self.means
        model_low_entropy.covars_ = self.covars

        # Initialize the HMM with high entropy
        model_high_entropy = hmm.GaussianHMM(
            n_components=self.n_hidden, covariance_type="diag"
        )
        model_high_entropy.startprob_ = self.startprob
        model_high_entropy.transmat_ = self.high_entropy_transitions
        model_high_entropy.means_ = self.means
        model_high_entropy.covars_ = self.covars

        # Calculate entropy
        entropy = -np.nansum(
            self.low_entropy_transitions[0] * np.log2(self.low_entropy_transitions[0])
        )
        entropy_h = -np.nansum(
            self.high_entropy_transitions[0] * np.log2(self.high_entropy_transitions[0])
        )
        print("Entropy of good data:", entropy)
        print("Entropy of bad data:", entropy_h)

        # Generate sequences
        num_chains = int(samples/2)
        chain_length = chain_length

        observed_sequences_low = []
        observed_sequences_high = []

        for _ in range(num_chains):
            obs_seq_low, _ = model_low_entropy.sample(chain_length)
            observed_sequences_low.append(obs_seq_low)
            obs_seq_high, _ = model_high_entropy.sample(chain_length)
            observed_sequences_high.append(obs_seq_high)

        # Convert lists to NumPy arrays
        observed_sequences_low = np.array(observed_sequences_low)
        y_low = np.zeros(len(observed_sequences_low))
        observed_sequences_high = np.array(observed_sequences_high)
        y_high = np.ones(len(observed_sequences_low))

        # concatenate 2 categories and shuffle
        y = np.concatenate([y_low, y_high], axis=0)
        observations = np.concatenate(
            [observed_sequences_low, observed_sequences_high], axis=0
        )
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        observations = observations[indices]
        labels = y[indices]

        # Save to HDF5
        with h5py.File("hmm_gaussian_chains.h5", "w") as f:
            f.create_dataset("label", data=labels)
            f.create_dataset("observed", data=observations)


if __name__ == "__main__":
    generator = BinaryHMMGenerator()
    generator.gen_binary_hmm()
