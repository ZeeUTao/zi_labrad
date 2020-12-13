import numpy as np
import logging


def load_csv(csv_file):
    data = np.loadtxt(csv_file, delimiter=',', dtype=float)
    return data


class zero_table(object):
    """
    A zero calibration table for given device
    """
    def __init__(self, device_name, awg_index):
        if device_name[:3] != 'dev':
            raise ValueError('device_name prefix is dev', device_name[:3])
        self.device_name = device_name
        self.awg_index = awg_index

        # carrier frequency of microwave source
        self.carrier_freq = None # GHz
        # optimized offset for the two ports
        self.offsets_I = None
        self.offsets_Q = None

        self.is_data_loaded = False

    @property
    def name(self):
        return (self.device_name, self.awg_index)

    def load_data(self, data):
        self.carrier_freq = data[:, 0]
        self.offsets_I = data[:, 1]
        self.offsets_Q = data[:, 2]
        self.is_data_loaded = True

    def get_offset(self, freq):
        if freq is None:
            return
        if self.is_data_loaded:
            _I = np.interp(freq, self.carrier_freq, self.offsets_I)
            _Q = np.interp(freq, self.carrier_freq, self.offsets_Q)
            return _I, _Q
        else:
            logging.info(
                "skip zero correction, no loaded data")
            # no correction
            return None


class Zero_correction(object):
    """zero correction for HDAWG
    We directly tuning the I and Q offsets, which is also given in
    labone UI.
    """
    def __init__(self, daq, registry_calibration):
        # zhinst.ziPython.ziDAQServer
        # typically passed from qubitContext
        self.registry_calibration = registry_calibration
        self.daq = daq
        self._device_dict = None
        self.valid_port = [(1, 2), (3, 4), (5, 6), (7, 8)]
        self.dict_tables = {}

    @property
    def device_dict(self):
        return self._device_dict

    @device_dict.setter
    def device_dict(self, _dict):
        """
        _dict：for example, {'hd_1':'dev8334','qa_1':'dev2592'}
        It is assign by
        dict(deviceInfo['ziQA_id'] + deviceInfo['ziHD_id'])
        """
        self._device_dict = _dict

    def _load_csv_data(self):
        for table in self.dict_tables.values():
            reg = self.registry_calibration.copy()
            reg_des = reg[table.device_name]['awg'+str(table.awg_index)]
            csv_file = reg_des.get('zero_data')
            if csv_file is None:
                continue
            data = load_csv(csv_file)
            table.load_data(data)

    def _add_table(self, device_idx):
        if device_idx[:2] == 'qa':
            device_name = self.device_dict[device_idx]
            table = zero_table(device_name, 0)
            self.dict_tables[table.name] = table
        elif device_idx[:2] == 'hd':
            device_name = self.device_dict[device_idx]
            for i in range(4):
                table = zero_table(device_name, i)
                self.dict_tables[table.name] = table

    def _create_registry_folder(self):
        reg = self.registry_calibration
        for device_idx, device_name in self.device_dict.items():
            if device_name not in reg._get_list()[0]:
                reg._subdir(device_name)
            if device_idx[:2] == 'qa':
                reg[device_name]._subdir('awg0')
            elif device_idx[:2] == 'hd':
                for i in range(4):
                    reg[device_name]._subdir('awg'+str(i))


    def init_tables(self, device_dict):
        """create zero_table objects, folders in registry
        and load data
        """
        self.device_dict = device_dict
        self._create_registry_folder()

        for device_idx in self.device_dict.keys():
            self._add_table(device_idx)
        self._load_csv_data()


    def get_table_name(self, qubit):
        """
        get_table_name from qubit
        For example, channels =
        [('xy_I', ('hd_1', 5)), ('xy_Q', ('hd_1', 6)),
         ('dc', ('hd_1', 7)), ('z', ('hd_1', 8))]
        """
        channels = qubit['channels']
        channels_dict = dict(channels)
        device_name = self.device_dict.get(channels_dict['xy_I'][0])
        port_I = int(channels_dict['xy_I'][1])
        port_Q = int(channels_dict['xy_Q'][1])
        if (port_I, port_Q) not in self.valid_port:
            raise ValueError('port_I, port_Q are not in', self.valid_port)
        awg_index = port_Q//2 - 1
        return (device_name, awg_index)

    def correct_xy(self, qubit):
        table_name = self.get_table_name(qubit)
        table = self.dict_tables.get(table_name)
        if table is None:
            logging.info(
                "skip zero correction, no table")
            return
        fc = qubit.get('xy_mw_fc')
        if type(fc) is not (int or float):
            # with unit
            fc = fc['GHz']
        offset = table.get_offset(fc)
        if offset is None:
            return
        dc_I, dc_Q = offset
        self.daq.setDouble('/{:s}/sigouts/{:d}/offset'.format(
            table.device_name, table.awg_index*2), dc_I)
        self.daq.setDouble('/{:s}/sigouts/{:d}/offset'.format(
            table.device_name, table.awg_index*2+1), dc_Q)
