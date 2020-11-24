import pytest
import numpy as np
from zilabrad.instrument.corrector import correct


zero_cor = correct.Zero_correction()


def example_zero_data():
    freqs = np.arange(4, 6, 0.1)
    data = np.random.random((len(freqs), 3))
    data[:, 0] = freqs
    return data


def test_get_IQ_ports():
    channels = [
        ('xy_I', ('hd_1', 5)), ('xy_Q', ('hd_1', 6)),
        ('dc', ('hd_1', 7)), ('z', ('hd_1', 8))]
    qubit = {}
    qubit['channels'] = channels
    zero_cor.deviceDict = {'hd_1': 'dev8334'}
    device_name, ports = zero_cor.get_IQ_ports(qubit)
    assert (device_name, ports) == ('dev8334', (5, 6))
    return


def test_get_offset():
    example_data = example_zero_data()
    zero_cor.load_data(example_data)
    _I, _Q = zero_cor.get_offset(np.random.random())
    return
