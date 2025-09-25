import numpy as np

class MemoryAdExNeuron:
    #    def __init__(self, dt=0.1):
    def __init__(self, dt=1e-3, sim_time=1000e-3):
        # Time step
        self.dt = dt
        self.timesteps = int(sim_time / dt)

        # Constants (taken from Neuronal Dynamics defaults)
        self.C = 200e-12  # Membrane capacitance
        self.gL = 10e-9  # Leak conductance
        self.EL = -70e-3  # Resting potential
        self.VT = -5e-3  # Rheobase threshold
        self.DT = 2e-3  # Slope factor
        self.tau_w = 8e-1  # Adaptation time constant
        self.a = 0.9e-9  # Subthreshold adaptation
        self.b = 7e-11  # Spike-triggered adaptation
        self.input_adaptation = 1e-8

        self.V_reset = -65e-3  # Reset potential
        self.V_spike = 0.01  # Spike height
        self.w = 0.63e-9  # 1e-9

        self.reset()

    def reset(self):
        self.V = self.EL
        self.spike_count = 0
        self.w = 0.63e-9

    #    def step(self, I_input):
    def simulate(self, input_current):
        self.reset()
        discretized_input = input_current * self.timesteps
        for i in range(self.timesteps):

            # Update membrane potential using AdEx dynamics
            dV = (-self.gL * (self.V - self.EL)
                  + self.gL * self.DT * np.exp((self.V - self.VT) / self.DT)
                  + input_current * 1e-9 / self.timesteps
                  + self.w) / self.C  # +w instead of -w

            # Update adaptation variable w
            # dw = ( self.a * (self.V - self.EL) - self.w + self.input_adaptation * input_current/self.timesteps) / self.tau_w
            # dw = (self.input_adaptation * input_current) / self.tau_w

            # Euler integration
            self.V += self.dt * dV
            # self.w += self.dt * dw

            # Spike condition
            if self.V >= self.V_spike:
                self.V = self.V_reset
                # self.w += self.b
                self.spike_count += 1

            discretized_input -= 1
        dw = (self.input_adaptation * input_current) / self.tau_w
        self.w += self.dt * dw * 0.001
        return self.spike_count

    def simulate_spike_train(self, input_array, reset=False):
        #if reset:
        #    self.reset()
        spike_count = 0
        output_spike_train = np.zeros(input_array.shape)
        for i,current in enumerate(input_array):
            dV = (-self.gL * (self.V - self.EL)
                  + self.gL * self.DT * np.exp((self.V - self.VT) / self.DT)
                  + current * 1e-9
                  + self.w) / self.C  # Add current directly

            self.V += self.dt * dV * 5

            if self.V >= self.V_spike:
                self.V = self.V_reset
                spike_count += 1
                output_spike_train[i] = 1

            # Update adaptation
            dw = (self.input_adaptation * current) / self.tau_w
            self.w += self.dt * dw * 0.01
        return output_spike_train

"""
# Example simulation
inputs = [0.3e-6] * 1000  # Input in Amps
neuron = MemoryAdExNeuron()
output_spikes = []
ws = []
for i, current in enumerate(inputs):
    spikes = neuron.simulate(current)
    ws.append(neuron.w)
    output_spikes.append(spikes)
    print(f"Input {i}: {current * 1e9:.2f} nA → Output spikes: {spikes}")

plt.plot(output_spikes)
plt.plot(ws)
plt.show()
"""