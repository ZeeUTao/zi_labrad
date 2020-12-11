"""
Resources for qubit and devices
"""
from zilabrad.pyle.workflow import switchSession
from zilabrad.pyle.registry import RegistryWrapper

from zilabrad.instrument.zurichHelper import zurich_qa, zurich_hd, ziDAQ
from zilabrad.instrument.servers.anritsu_MG3692C import AnritsuServer
from zilabrad.util import singleton
from zilabrad.instrument.corrector import correct

import labrad


def update_session(user='hwh'):
    cxn = labrad.connect()
    ss = switchSession(cxn, user=user, session=None)
    return ss


def loadQubits(sample, write_access=False):
    """Get local copies of the sample configuration stored in the
    labrad.registry.

    If you do not use labrad, you can create a class as a wrapped dictionary,
    which is also saved as files in your computer.
    The sample object can also read, write and update the files

    Returns the local sample config, and also extracts the individual
    qubit configurations, as specified by the sample['config'] list.  If
    write_access is True, also returns the qubit registry wrappers themselves,
    so that updates can be saved back into the registry.
    """
    Qubits = [sample[q] for q in sample['config']]
    sample = sample.copy()
    qubits = [sample[q] for q in sample['config']]

    # only return original qubit objects if requested
    if write_access:
        return sample, qubits, Qubits
    else:
        return sample, qubits


@singleton
class qubitContext(object):
    """
    resources of the experimental devices
    """

    def __init__(self, cxn=None):
        """
        Args:
            deviceInfo (dict): devices dictionary
            servers_xxx: object of server
        """
        self.cxn = cxn
        if self.cxn is None:
            self.cxn = labrad.connect()

        self.deviceInfo = self.loadInfo(paths=['Servers', 'devices'])
        self.wiring = dict(self.deviceInfo.get('wiring'))

        self.IPdict_microwave = dict(self.deviceInfo['microwave_source'])
        self.servers_microwave = AnritsuServer(
            list(self.IPdict_microwave.values())
            )
        self.servers_qa = self._get_zurich_servers('ziQA_id')
        self.servers_hd = self._get_zurich_servers('ziHD_id')
        self.servers_daq = ziDAQ().daq

        self.registry_calibration = RegistryWrapper(
            self.cxn, ['', 'Zurich Calibration'])
        self.init_correct()

    @property
    def ADC_FS(self):
        """ADC sampling rate
        """
        qa = self.get_server('qa', 'qa_1')
        return qa.FS

    @property
    def DAC_FS(self):
        """DAC sampling rate
        """
        hd = self.get_server('hd', 'hd_1')
        return hd.FS

    def get_server(self, type: str, name: str or None):
        """return the object of server
        example: get_server('qa', 'qa_1')
        """
        if type == 'qa':
            return self.servers_qa[name]
        elif type == 'hd':
            return self.servers_hd[name]
        elif type == 'daq':
            return self.servers_daq
        elif type == 'microwave_source':
            return self.servers_microwave

    def get_servers_group(self, type='qa'):
        """return a collection of servers for the same type
        (if they have more than ones)
        """
        if type == 'qa':
            return self.servers_qa
        elif type == 'hd':
            return self.servers_hd
        elif type == 'zurich':
            return {
                **self.servers_qa,
                **self.servers_hd
            }

    def init_correct(self):
        self.device_mapping_dict = dict(self.deviceInfo['ziQA_id'] +
                                        self.deviceInfo['ziHD_id'])
        # example: {'qa_1':'dev2591','hd_1':'dev8334'}
        self.zero_correction = correct.Zero_correction(
            self.servers_daq, self.registry_calibration)
        self.zero_correction.init_tables(self.device_mapping_dict)

    @staticmethod
    def getQubits_paras(qubits: dict, key: str):
        """ Get specified parameters from dictionary (qubits)
        """
        return [_qubit[key] for _qubit in qubits]

    def loadInfo(self, paths):
        """
        load the sample information from specified directory.

        Args:
            paths (list): Array with data of waveform 1.

        Returns:
            reg.copy() (dict):
            the key-value information from the directory of paths
        """
        reg = RegistryWrapper(self.cxn, ['']+paths)
        return reg.copy()

    def _get_zurich_servers(self, name: str):
        """
        Args:
            name: deviceInfo key, 'ziQA_id', 'ziHD_id'
        Returns (dict):
            for example {'1',object}, object is an
            instance of device server (class).
            Do not worry about the instance of the same device is
            recreated, which is set into a conditional singleton.
        """
        self._server_class = {
            'ziQA_id': zurich_qa,
            'ziHD_id': zurich_hd,
        }

        if name not in self._server_class:
            raise TypeError("No such device type %s" % (name))
        deviceDict = dict(self.deviceInfo[name])

        for _id in deviceDict:
            server = self._server_class[name]
            serversDict = server(
                _id, device_id=deviceDict[_id],
                labone_ip=self.deviceInfo['labone_ip'])
            # here "=" is not an error, because the
            # class is a singletonMany (returns a dict of objects)
        return serversDict

    def getPorts(self, qubits):
        """
        Get the AWG ports for zurich HD according to whether the
        corresponding keys exist.
        Args:
            qubits, list of dictionary
        Returns:
            ports (list)
            for example, [('hd_1', 7), ('hd_1', 5), ('hd_1', 6), ('hd_1', 8)]

        TODO:
        1. ports dictionary should only be created once in
        the beginning of experiment.
        """
        ports = []
        for q in qubits:
            channels = dict(q['channels'])
            # the order must be 'dc,xy,z'! match the order in qubitServer
            if 'dc' in q.keys():
                # if channels have not 'dc', ports += [None]
                ports += [channels.get('dc')]
            if 'xy' in q.keys():
                ports += [channels['xy_I'], channels['xy_Q']]
            if 'z' in q.keys():
                ports += [channels.get('z')]
        return ports

    def clearTempParas(self):
        attr_names = ['ports']
        for name in attr_names:
            if hasattr(self, name):
                delattr(self, name)

    def refresh(self):
        self.deviceInfo = self.loadInfo(paths=['Servers', 'devices'])
        self.wiring = dict(self.deviceInfo.get('wiring'))

        self.servers_qa = self._get_zurich_servers('ziQA_id')
        self.servers_hd = self._get_zurich_servers('ziHD_id')
