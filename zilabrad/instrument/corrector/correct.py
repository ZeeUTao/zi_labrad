import numpy as np
import labrad


def load_csv(csv_file):
    data = np.loadtxt(csv_file, delimiter=',', dtype=float)
    return data


class Zero_correction(object):
    """
    zero correction can be directly implemented by tuning the offsets,
    which is also given in labone UI.
    """
    def __init__(self):
        self.cxn = None
        self.deviceDict = None
        # self.load_device_dict()

        self.freq = None
        # frequency of microwave source
        self.offsets_I = None
        self.offsets_Q = None
        # optimized offset for the two ports

    def load_device_dict(self):
        self.cxn = labrad.connect()
        reg = RegistryWrapper(self.cxn, ['', 'Servers', 'devices'])
        deviceInfo = reg.copy()
        self.deviceDict = dict(deviceInfo['ziQA_id'] + deviceInfo['ziHD_id'])

    def get_IQ_ports(self, qubit):
        """
        For example, channels =
        [('xy_I', ('hd_1', 5)), ('xy_Q', ('hd_1', 6)), ('dc', ('hd_1', 7)),
         ('z', ('hd_1', 8))]
        """
        channels = qubit['channels']
        channels_dict = dict(channels)
        device_name = self.deviceDict.get(channels_dict['xy_I'][0])
        if device_name[:3] != 'dev':
            raise ValueError('device_name prefix is dev', device[:3])

        port_I = int(channels_dict['xy_I'][1])
        port_Q = int(channels_dict['xy_Q'][1])
        return (device_name, (port_I, port_Q))

    def load_data(self, data):
        self.freq = data[:, 0]
        self.offsets_I = data[:, 1]
        self.offsets_Q = data[:, 2]

    def get_offset(self, freq):
        _I = np.interp(freq, self.freq, self.offsets_I)
        _Q = np.interp(freq, self.freq, self.offsets_Q)
        return _I, _Q

    def set_offset(self, var_I, var_Q):
        return
