from zilabrad.pyle.workflow import switchSession
from zilabrad.pyle.registry import RegistryWrapper

from zilabrad.instrument.zurichHelper import zurich_qa, zurich_hd
from zilabrad.util import singleton


import labrad


def update_session(user='hwh'):
    cxn=labrad.connect()
    ss = switchSession(cxn,user=user,session=None)
    return ss
    
    
    

def loadQubits(sample, write_access=False):
    """Get local copies of the sample configuration stored in the labrad.registry.
    
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
    create a singleton object to provides servers, information about the experimental devices
    """
    def __init__(self,cxn = None):
        """
        Args:
            deviceInfo (dict): devices dictionary
            servers_xxx: object of server
        """
        self.cxn = cxn
        if self.cxn is None:
            self.cxn = labrad.connect()
        self._server_class = {
        'ziQA_id':zurich_qa,
        'ziHD_id':zurich_hd,
        }

        self.deviceInfo = self.loadInfo(paths=['Servers','devices']) 
        self.servers_qa = self.get_servers('ziQA_id')
        self.servers_hd = self.get_servers('ziHD_id')
        #self.servers_microwave = self.get_microwaveServer()
        self.IPdict_microwave = self.deviceInfo['microwave_source']



    def loadInfo(self,paths):
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


    def get_servers(self,name: str):
        """
        Args: 
            name: deviceInfo key, 'ziQA_id', 'ziHD_id'
        Returns:
            a dictionary, for example {'1',object}, object is an instance of device server (class)
            Do not worry about the instance of the same device is recreated, which is set into a conditional singleton.
        """

        if name not in self._server_class:
            raise TypeError("No such device type %s"%(name))
        deviceDict = dict(self.deviceInfo[name])

        for _id in deviceDict:
            server = self._server_class[name]
            serversDict = server(_id,device_id=deviceDict[_id],labone_ip=self.deviceInfo['labone_ip'])
            # here "=" is not an error, because the class is a singletonMany (returns a dict of objects)
        return serversDict

    def get_microwaveServer(self):
        """
        usually return anritsu_server
        """
        name = str(self.deviceInfo['microwave_server'])
        server = self.cxn[name]
        return server

    def refresh(self):
        self.deviceInfo = self.loadInfo(paths=['Servers','devices']) 
        self.servers_qa = self.get_servers('ziQA_id')
        self.servers_hd = self.get_servers('ziHD_id')
        #self.servers_microwave = self.get_microwaveServer()
        self.IPdict_microwave = self.deviceInfo['microwave_source']