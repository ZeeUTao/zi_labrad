from zilabrad.pyle.workflow import switchSession
from zilabrad.pyle.registry import RegistryWrapper
import labrad


def update_session(user='hwh'):
    cxn=labrad.connect()
    ss = switchSession(cxn,user=user,session=None)
    return ss
    
    
    
def loadInfo(paths=['Servers','devices']):
    """
    load the sample information from specified directory.

    Args:
        paths (list): Array with data of waveform 1.

    Returns: 
        reg.copy() (dict): 
        the key-value information from the directory of paths

    ** waveform reload needs each two channel one by one.
    """
    cxn=labrad.connect()
    reg = RegistryWrapper(cxn, ['']+paths)
    return reg.copy()
    

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