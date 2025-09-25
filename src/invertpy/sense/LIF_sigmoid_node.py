import numpy as np

class LIFNeuron:
    def __init__(self, tau=5.0, R=1.0, dt=1.0, V_threshold=0.1, V_reset=0.0):
        self.tau = tau  # Membrane time constant (ms)
        self.R = R  # Membrane resistance
        self.dt = dt  # Time step
        self.V = 0.0  # Membrane potential
        self.V_threshold = V_threshold
        self.V_reset = V_reset

    def step(self, input_spike):
        # Convert spike (0 or 1) to input current (scaled)
        input_current = input_spike * 1.0  # input weight is 1 here, can be changed

        # Update membrane potential with leak and input
        dV = (-self.V + self.R * input_current) * (self.dt / self.tau)
        self.V += dV

        # Check for spike: if voltage crosses threshold
        if self.V >= self.V_threshold:
            output_spike = 1
            self.V = self.V_reset  # reset potential
        else:
            output_spike = 0
        return output_spike

    def steps(self, input_spikes):
        output_spikes = []
        for spike in input_spikes:
            out = self.step(spike)
            output_spikes.append(out)
        output_spikes = np.array(output_spikes)
        return output_spikes

# Test behaviour
# for i in [0.05,0.1,0.35,0.67,0.98]:
#     neuron = LIFNeuron()
#     input_spikes = np.random.binomial(1, i, 1000)  # ~5% spike rate
#     output_spikes = neuron.steps(input_spikes)
#     print(i*1000,output_spikes.sum())