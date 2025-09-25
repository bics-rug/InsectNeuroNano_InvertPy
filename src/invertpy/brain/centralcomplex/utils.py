import numpy as np

class CW_Ring:
    def __init__(self, n, bias_strength=2.0, self_excitation=1.0):
        self.n = n
        self.state = np.zeros(n, dtype=int)
        self.state[0] = 1  # start with neuron 0 active
        self.Wsym = self.build_symmetric_weights(self_excitation)
        self.Wasym = self.build_asymmetric_weights(bias_strength)

    def build_symmetric_weights(self, self_excitation):
        W = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i==j:
                    W[i, j] = self_excitation # self-excitation (stabilize one-hot)
        return W

    def build_asymmetric_weights(self, bias_strength):
        W = np.zeros((self.n, self.n))
        for i in range(self.n):
            W[i, (i + 1) % self.n] = bias_strength    # excite next (CW)
        return W

    def step(self, input_vector):
        """
        input_vector: array of shape (n,), one value per neuron
        """
        input_vector = np.asarray(input_vector)
        assert input_vector.shape == (self.n,), "Input vector must be shape (n,)"

        # Scale each row of Wasym by the corresponding input value
        Winput = self.Wasym * input_vector[:, np.newaxis]

        # Compute the total input current
        input_current = self.state @ (self.Wsym + Winput)

        # Winner-take-all
        new_state = np.zeros(self.n, dtype=int)
        new_state[np.argmax(input_current)] = 1
        self.state = new_state
        return self.state

class CCW_Ring:
    def __init__(self, n, bias_strength=2.0, self_excitation=1.0):
        self.n = n
        self.state = np.zeros(n, dtype=int)
        self.state[0] = 1  # start with neuron 0 active
        self.Wsym = self.build_symmetric_weights(self_excitation)
        self.Wasym = self.build_asymmetric_weights(bias_strength)

    def build_symmetric_weights(self, self_excitation):
        W = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i==j:
                    W[i, j] = self_excitation # self-excitation (stabilize one-hot)
        return W

    def build_asymmetric_weights(self, bias_strength):
        W = np.zeros((self.n, self.n))
        for i in range(self.n):
            W[i, (i - 1) % self.n] = bias_strength    # excite next (CW)
        return W

    def step(self, input_vector):
        """
        input_vector: array of shape (n,), one value per neuron
        """
        input_vector = np.asarray(input_vector)
        assert input_vector.shape == (self.n,), "Input vector must be shape (n,)"

        # Scale each row of Wasym by the corresponding input value
        Winput = self.Wasym * input_vector[:, np.newaxis]

        # Compute the total input current
        input_current = self.state @ (self.Wsym + Winput)

        # Winner-take-all
        new_state = np.zeros(self.n, dtype=int)
        new_state[np.argmax(input_current)] = 1
        self.state = new_state
        return self.state

