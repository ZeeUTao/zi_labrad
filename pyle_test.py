from pyle import sweeps
import labrad
import time
import numpy as np
from pyle.util import sweeptools as st
from pyle.workflow import switchSession
from pyle.pipeline import returnValue, FutureList
from pyle.sweeps import checkAbort

from labrad.units import Unit,Value
V, mV, us, ns, GHz, MHz, dBm, rad = [Unit(s) for s in ('V', 'mV', 'us', 'ns', 'GHz', 'MHz', 'dBm', 'rad')]

from importlib import reload

ar = st.r
cxn=labrad.connect()
dv = cxn.data_vault

# specify the sample, in registry   
ss = switchSession(cxn,user='hwh',session=None)  


    
def loadQubits(sample, write_access=False):
    """Get local copies of the sample configuration stored in the registry.
    
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



def gridSweep(axes):
    # yield (all axes), (swept axes)
    if not len(axes):
        yield (), ()
    else:
        (param, _label), rest = axes[0], axes[1:]
        if np.iterable(param): # TODO: different way to detect if something should be swept
            for val in param:
                for all, swept in gridSweep(rest):
                    yield (val,) + all, (val,) + swept
        else:
            for all, swept in gridSweep(rest):
                yield (param,) + all, swept

def dataset_create(dv,dataset):
    """Create the dataset."""
    dv.cd(dataset.path, dataset.mkdir)
    print(dataset.dependents)
    dv.new(dataset.name, dataset.independents, dataset.dependents)
    if len(dataset.params):
        dv.add_parameters(tuple(dataset.params))


        
def s21_scan(sample,freq = ar[1.:6.:0.01,GHz],power = st.r[-1:-30:1,dBm],zpa =0.0,name = 'S21',des='',
    measure=0,stats = 300,save=True):
    
    # load qubits 
    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    q = qubits[measure]
    
    # user input the parameters
    axes = [(freq, 'freq'),(power,'power'),(zpa,'zpa')]
    deps = []
    deps.append(('I','',''))
    deps.append(('Q','',''))
    deps.append(('amp','',''))
    deps.append(('phase','',''))

    kw = {'stats': stats}
    
    # create dataset
    dataset = sweeps.prepDataset(sample, name+des, axes, deps,measure=measure, kw=kw)
    dataset_create(dv,dataset)
    
    indeps = dataset.independents
    
    print('11111')
    def runQ(x):
        """ virtual_adc, pretend running 
        """
        time.sleep(0.01)
        I,Q = np.sin(10*x)+0.2,np.cos(10*x)
        return np.array([I,Q])  

    def transUnit(args):
        args_new = []
        for a in args:
            if type(a) is not Value:
                args_new.append(a)
            else:
                args_new.append(a[a.unit])      
        return args_new
    
    axes_scans = checkAbort(gridSweep(axes), prefix=[1])
    for axes_scan in axes_scans:
        (freq,power,zpa) = axes_scan[0]
        
        data = runQ(freq[freq.unit])
        I = data[0]
        Q = data[1]
        amp = np.abs(I+1j*Q)
        phase = np.angle(I+1j*Q)
        data_deps = [I,Q,amp,phase]
        
        indeps_scan = axes_scan[1]
        indeps_scan = transUnit(indeps_scan)
        
        data_send = indeps_scan + data_deps
        
        print(data_send)
        dv.add(data_send)
    
    
    return

if __name__ == '__main__':
    s21_scan(ss)