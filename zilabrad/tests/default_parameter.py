"""
default parameters in Registry
"""
from zilabrad.pyle.registry import RegistryWrapper
import labrad
from labrad.units import Value


_devices = {
    'labone_ip': 'localhost',
    'microwave_server': 'anritsu_server',
    'microwave_source': [
        ('anritsu_r_1', 'TCPIP0::192.168.1.241::inst0::INSTR'),
        ('anritsu_xy_1', 'TCPIP0::192.168.1.240::inst0::INSTR')],
    'wiring': [('qa_1', 7)],
    'ziHD_id': [('hd_1', 'dev8334')],
    'ziQA_id': [('qa_1', 'dev2591')]
}

_qubit_para = {
    'awgs_pulse_len': Value(1.0, 'us'),
    'bias': Value(0.0, 'V'),
    'bias_end': Value(1.0, 'us'),
    'bias_start': Value(1.0, 'us'),
    'center|0>': [0.0, 0.0],
    'center|1>': [2.0, 2.0],
    'channels': [('xy_I', ('hd_1', 1)),
                 ('xy_Q', ('hd_1', 2)),
                 ('dc', ('hd_1', 3)),
                 ('z', ('hd_1', 4))],
    'demod_freq': 20000000.0,
    'demod_phase': 0.0,
    'f10': Value(6.9145, 'GHz'),
    'piAmp': 0.76,
    'piLen': Value(40.0, 'ns'),
    'qa_adjusted_phase': Value(0.0, 'MHz'),
    'qa_start_delay': Value(85.0, 'ns'),
    'readout_amp': Value(-15.0, 'dBm'),
    'readout_delay': Value(197.8, 'ns'),
    'readout_freq': Value(5.1795, 'GHz'),
    'readout_len': Value(2.0, 'us'),
    'readout_mw_fc': Value(5.28, 'GHz'),
    'readout_mw_power': Value(18.0, 'dBm'),
    'stats': 1024,
    'xy_mw_fc': Value(6.4, 'GHz'),
    'xy_mw_power': Value(18.0, 'dBm'),
    'zpa': Value(0.0, 'V')
}


def set_devices():
    cxn = labrad.connect()

    reg = RegistryWrapper(cxn, [''])
    reg._subdir('Servers')
    reg_server = reg['Servers']

    reg_server._subdir('devices')
    reg_server['devices'] = _devices
    return


def set_user():
    cxn = labrad.connect()
    reg = RegistryWrapper(cxn, [''])

    reg_user = reg._subdir('user_example')

    reg_user['sample'] = ['sample1']

    reg_user._subdir('sample1')
    reg_user['sample1']['config'] = ['q1']

    reg_user['sample1']._subdir('q1')
    reg_user['sample1']['q1'] = _qubit_para
    return


def main():
    set_devices()
    set_user()


if __name__ == "__main__":
    _continue = input("create default parameter\
(Enter 1 to continue)") or 0
    if _continue == '1':
        main()
