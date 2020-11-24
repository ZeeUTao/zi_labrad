import pytest
import numpy as np
from zilabrad.instrument.corrector import correct


class daq_dummy:
    def setDouble(self, path, value):
        print('setDouble', path, value)
        return


def example_zero_data():
    freqs = [4., 5., 6.]
    data = np.zeros((len(freqs), 3))
    data[:, 0] = freqs
    data[:, 1] = [0.1, 0.2, 0.3]
    data[:, 2] = [0.4, 0.5, 0.6]
    return data


daq = daq_dummy()
zero_cor = correct.Zero_correction(daq)
_data1 = example_zero_data()


def test_zero_table():
    table = correct.zero_table('dev8334', 1)
    assert table.name == ('dev8334', 1)
    assert table.awg_index == 1
    assert table.device_name == 'dev8334'

    assert table.is_data_loaded is False
    assert table.get_offset(5.0) is None
    table.load_data(_data1)
    assert table.is_data_loaded is True
    offsets = table.get_offset(5.0)
    assert offsets == (0.2, 0.5)


def test_correct_xy():
    channels = [
        ('xy_I', ('hd_1', 5)), ('xy_Q', ('hd_1', 6)),
        ('dc', ('hd_1', 7)), ('z', ('hd_1', 8))]
    qubit = {}
    qubit['channels'] = channels
    zero_cor.device_dict = {'hd_1': 'dev8334'}
    device_name, ports = zero_cor.get_table_name(qubit)

    zero_cor.add_table(qubit)
    table = zero_cor.dict_tables[(device_name, ports)]
    table.load_data(_data1)
    zero_cor.correct_xy(qubit)
    qubit['xy_mw_fc'] = 5.0
    zero_cor.correct_xy(qubit)
    return
