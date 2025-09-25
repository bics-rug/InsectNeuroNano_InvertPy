import sys
sys.path.append('../src')  # Add path to where invertpy is located

from invertpy.sense.polarisation import MinimalDevicePolarisationSensor
from adex_memory import MemoryAdExNeuron
import matplotlib.pyplot as plt
import numpy as np

input_length = 1000
# Run the AdEx neuron with the spike train
neuron = MemoryAdExNeuron()
output_spike_rates,input_spike_rates = [],[]
for i in range(850):
    # Try manual input
    if i < 300:
        input_current = np.zeros(input_length)
        input_current[200:800] = 0.0115  # 0.0115 is the weight between D & M
        input_spike_rates.append(0.6)
    else:
        input_current = np.zeros(input_length)
        input_current[0:300] = 0.0115  # 0.0115 is the weight between D & M
        input_spike_rates.append(0.3)

    # Use input from output of D node
    #input_current = spike_train
    print('here',neuron.w)
    output_spike_train = neuron.simulate_spike_train(input_current)
    output_spike_rate = output_spike_train.sum()/input_length
    output_spike_rates.append(output_spike_rate)
    # Step 5: Print and plot result
    print(f"\nAdEx neuron output spikes: {output_spike_rate}")

# Transform output spike counts into spike rates
output_spike_rates = np.array(output_spike_rates)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(input_spike_rates,label='input')
#plt.title("Current Input to AdEx Neuron (not from D-Node Spike Train)")
ax1.set_xlabel("Insect step")
ax1.set_ylabel("Spike rate")
ax1.grid(True)
#plt.tight_layout()
#plt.show()

ax2.plot(output_spike_rates,label='output',c='orange')
plt.title("AdEx M node input & output spikes")
ax2.set_xlabel("Insect step")
ax2.set_ylabel("Spike rate")
plt.grid(True)

ax1.legend(loc='upper left')
ax2.legend(loc='lower right')
plt.tight_layout()
plt.show()