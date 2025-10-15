import numpy as np
import matplotlib.pyplot as plt
from invertpy.sense.LIF import LIFNeuron

def diagonal_synapses(nb_in, nb_out, fill_value=1, tile=False, dtype='float32', bias=None):
    """
    Creates diagonal synapses.

    Examples
    --------
    # >>> diagonal_synapses(3, 4, fill_value=2)
    # >>> np.array([[2, 0, 0, 0],
    # >>>           [0, 2, 0, 0],
    # >>>           [0, 0, 2, 0]], dtype='float32')
    #
    # >>> diagonal_synapses(3, 6, tile=True, fill_value=1)
    # >>> np.array([[1, 0, 0, 1, 0, 0],
    # >>>           [0, 1, 0, 0, 1, 0],
    # >>>           [0, 0, 1, 0, 0, 1]], dtype='float32')

    Parameters
    ----------
    nb_in: int
        the number of the input units.
    nb_out: int
        the number of the output units.
    fill_value: float
        the value of the non-zero synaptic weights.
    tile: bool, optional
        if True and nb_in != nb_out, then it wraps the diagonal starting from the beginning.
    dtype: np.dtype | str
        the type of the values for the synaptic weights.
    bias: float, optional
        the value of all the biases. If None, no bias is returned.

    Returns
    -------
    params: np.ndarray | tuple
        the generated synaptic weights and the bias (if requested)
    """
    if tile:
        w = np.zeros((nb_in, nb_out), dtype=dtype)
        if nb_out < nb_in:
            _w = w
            _nb_in = nb_in
            _nb_out = nb_out
        else:
            _w = w.T
            _nb_in = nb_out
            _nb_out = nb_in

        i = 0
        while np.sum(~np.isclose(_w, 0)) < _nb_in:
            i_start = i * _nb_out
            i_end = np.minimum((i + 1) * _nb_out, _nb_in)
            _w[i_start:i_end] = fill_value * np.eye(i_end - i_start, _nb_out)
            i += 1

        if nb_out < nb_in:
            w = _w
        else:
            w = _w.T
    else:
        w = fill_value * np.eye(nb_in, nb_out, dtype=dtype)
    if bias is None:
        return w
    else:
        return w, np.full(nb_out, fill_value=bias, dtype=dtype)

def stdp_synaptic_update(weight, pre_spikes, post_spikes, A_plus=1e-8, tau_plus=15,
                               ):
        pre_spike_times = np.where(pre_spikes == 1)[0]
        post_spike_times = np.where(post_spikes == 1)[0]

        # Create all pairwise delta_t
        delta_t = post_spike_times[:, None] - pre_spike_times[None, :]

        # Potentiation for delta_t > 0
        dwp = A_plus * np.exp(-delta_t / tau_plus) * (delta_t > 0)

        # Sum all updates
        delta_w = dwp.sum()
        # Update weight
        weight += delta_w
        return weight

constant_input = np.random.binomial(1, 0.5, 1000)
direction1 = np.random.binomial(1, 0.55, 1000)
direction2 = np.random.binomial(1, 0.36, 1000)
direction3 = np.random.binomial(1, 0.58, 1000)

w_dir2mem = diagonal_synapses(3, 3, fill_value=0.0115, dtype=np.float32)
adaptive_w_ct2mem1 = w_dir2mem[0][0]
adaptive_w_ct2mem2 = w_dir2mem[0][0]
adaptive_w_ct2mem3 = w_dir2mem[0][0]
M1 = LIFNeuron()
M2 = LIFNeuron()
M3 = LIFNeuron()

memory = []
n_steps = 100
for step in range(n_steps):
    post_spikes_1 = M1.steps(constant_input * adaptive_w_ct2mem1)
    post_spikes_2 = M2.steps(constant_input * adaptive_w_ct2mem2)
    post_spikes_3 = M3.steps(constant_input * adaptive_w_ct2mem3)
    adaptive_w_ct2mem1 = stdp_synaptic_update(adaptive_w_ct2mem1, direction1, post_spikes_1)
    adaptive_w_ct2mem2 = stdp_synaptic_update(adaptive_w_ct2mem2, direction2, post_spikes_2)
    adaptive_w_ct2mem3 = stdp_synaptic_update(adaptive_w_ct2mem3, direction3, post_spikes_3)
    memory.append([adaptive_w_ct2mem1, adaptive_w_ct2mem2, adaptive_w_ct2mem3])

memory = np.array(memory)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot((direction1.sum()/1000).repeat(n_steps),'--',label='D1')
ax1.plot((direction2.sum()/1000).repeat(n_steps),'--',label='D2')
ax1.plot((direction3.sum()/1000).repeat(n_steps),'--',label='D3')
ax2.plot(memory[:,0],label='M1')
ax2.plot(memory[:,1],label='M2')
ax2.plot(memory[:,2],label='M3')
ax1.set_xlabel('Step')
ax1.set_ylabel('Input spiking rate (D)')
ax2.set_ylabel('Weight (M)')
ax1.legend()
ax2.legend()
plt.show()
