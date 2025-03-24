import numpy as np

rng = np.random.RandomState(42)


class GenHMM:
    """
    A class used to generate Hidden Markov Models (HMMs).
    Attributes
    ----------
    initial_probs : array-like, optional
        Initial state probabilities vector.
    transitions : array-like, optional
        State transition probabilities matrix.
    emissions : array-like, optional
        Emission probabilities matrix.
    Methods
    -------
    __gen_probs_matrix(rows=2, cols=2)
        Generates a probabilities matrix with row values summing to 1.
    __gen_probs_vector(rows=2)
        Generates a probabilities vector with values summing to 1.
    gen_observables_options(observables=1, n_options=2)
        Generates sets of possible values for each observable.
    gen_hmm(samples, observables=1, hiddenStates=[0, 1], observablesOptions=None, initialProbs=None, transitions=None, emissions=None)
        Generates a Hidden Markov Model with specified parameters or random values if not provided.
    """

    def __init__(self, initial_probs=None, transitions=None, emissions=None):
        self.initial_probs = initial_probs
        self.transitions = transitions
        self.emissions = emissions

    # Function for generating probabilities matrix, with row values summing to 1
    def __gen_probs_matrix(rows=2, cols=2):
        A = rng.rand(rows, cols)
        rowSum = A.sum(axis=1, keepdims=True)
        normalizedA = A / rowSum
        return normalizedA

    # Function for generating probabilities vector, with values summing to 1
    def __gen_probs_vector(rows=2):
        A = rng.rand(rows)
        rowSum = A.sum()
        normalizedA = A / rowSum
        return normalizedA

    def gen_observables_options(observables=1, n_options=2):
        """The sets of values possible for each observable, are here set as integers"""
        observablesOptions = [[] for _ in range(observables)]
        for i in range(observables):
            for j in range(n_options):
                observablesOptions[i].append(j - 1)

    # Function generating Hidden Markov Chain
    def gen_hmm(
        self,
        samples,
        observables=1,
        hiddenStates=[0, 1],
        observablesOptions=None,
        initialProbs=None,
        transitions=None,
        emissions=None,
    ):

        # If not given as argument, probabilities for the first hidden state are set he randomly
        if initialProbs == None:
            initialProbs = self.__gen_probs_vector(len(hiddenStates))

        # If not given as argument, the transition matrix is set here randomly
        if transitions == None:
            transitions = self.__gen_probs_matrix(len(hiddenStates), len(hiddenStates))

        # If not given as argument, the emissions matrix is set here randomly
        if emissions == None:
            emissions = []
            for observable in range(observables):
                emissions.append(
                    self.__gen_probs_matrix(
                        len(hiddenStates), len(observablesOptions[observable])
                    )
                )

        # Initializing empty sets for the chains of hidden and observed states
        hiddenVariables = []
        observedVariables = [[] for _ in range(observables)]

        # Setting the initial hidden state
        currentState = rng.choice(hiddenStates, p=initialProbs)

        # Generating chains of hidden and observed states
        for i in range(samples):
            hiddenVariables.append(currentState)
            for obs_idx in range(observables):
                observedVariables[obs_idx].append(
                    rng.choice(
                        observablesOptions[obs_idx], p=emissions[obs_idx][currentState]
                    )
                )

            currentState = rng.choice(
                hiddenStates, p=transitions[hiddenVariables[i - 1]]
            )

        return hiddenVariables, observedVariables
