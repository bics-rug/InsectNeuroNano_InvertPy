import numpy as np

def generate_spike_trains(input, time_window=1000):
    """
    Generate spike trains from float inputs using rate-based Poisson encoding.

    Parameters:
    - inputs: float value between 0 and 1
    - time_window: how many time steps to simulate

    Returns:
    - spike_trains: list of 3 lists, each containing 0s and 1s
    """
    spike_trains = []

    # For each float value, generate a spike train
    spikes = (np.random.rand(time_window) < input).astype(int)
    spike_trains.append(spikes.tolist())

    return spike_trains
