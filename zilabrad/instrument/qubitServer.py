# -*- coding: utf-8 -*-
"""
qubitServer to control all instruments

Created on 2020.09.09 20:57
@author: Tao Ziyu
"""

from functools import wraps
import logging
import numpy as np 

from zilabrad.instrument.zurichHelper import _mpAwg_init
from zilabrad.instrument import waveforms

import labrad
from labrad.units import Unit,Value
_unitSpace = ('V','mV','us','ns','s','GHz','MHz','kHz','Hz','dBm','rad','None')
V, mV, us, ns,s, GHz, MHz,kHz,Hz, dBm, rad,_l  = [Unit(s) for s in _unitSpace]

cxn = labrad.connect()
dv = cxn.data_vault

np.set_printoptions(suppress=True)


_noisy_printData = True



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






def dataset_create(dataset,dv=dv):
    """Create the dataset. 
    see 
    dataset = sweeps.prepDataset(*args)
    dv = labrad.connect().dataVault
    dataVault script is in "/server/py3_data_vault"
    """
    dv.cd(dataset.path, dataset.mkdir)
    logging.info(dataset.dependents)
    logging.info(dataset.independents)
    dv.new(dataset.name, dataset.independents, dataset.dependents)
    if len(dataset.params):
        dv.add_parameters(tuple(dataset.params))


def Unit2SI(a):
    if type(a) is not Value:
        return a
    elif a.unit in ['GHz','MHz']:
        return a['Hz']
    elif a.unit in ['ns','us']:
        return a['s']
    else:
        return a[a.unit] 

def Unit2num(a):
    if type(a) is not Value:
        return a
    else:
        return a[a.unit] 

def RunAllExperiment(exp_devices,function,iterable,
                     collect: bool = True,
                     raw: bool = False):
    """ Define an abstract loop to iterate a funtion for iterable

    Example:
    prepare parameters: scanning axes, dependences and other parameters
    start experiment with special sequences according to the parameters
    
    Args:
        exp_devices: (list/tuple): instances used to control device
        function: give special sequences, parameters, and start exp_devices, that are to be called by run()
        iterable: iterated over to produce values that are fed to function as parameters.
        collect: if True, collect the result into an array and return it; else, return an empty list
    """

    def run(function, paras):
        # pass in all_paras to the function
        all_paras = [Unit2SI(a) for a in paras[0]]
        swept_paras = [Unit2num(a) for a in paras[1]]
        result = function(exp_devices,all_paras)
        if raw:
            result_raws = np.asarray(result)
            for result_raw in result_raws.T:
                dv.add(result_raw)
            return result_raws
        else:
            data_send = list(swept_paras) + list(result)
            dv.add(data_send)

        if _noisy_printData == True:
            print(
                str(np.round(data_send,4))
                )      
        return result

    
    result_list = []    
    for paras in iterable:
        result = run(function,paras)
        if collect:
            result_list.append(result)
    result_list = np.asarray(result_list)
    return result_list



## qubit Mapping
## Get specified parameters from dictionary (qubits)

def getQubits_paras(qubits: dict, key: str):
    return [_qubit[key] for _qubit in qubits]



        
def getQubits_awgPort(qubits):
    """
    Get the AWG ports for zurich HD according to whether the corresponding keys exist
    Args: qubits, list of dictionary
    Returns: ports, list
    """
    ports = []
    for q in qubits:
        if 'dc' in q.keys():
            ports += [q.channels['dc']]
        if 'xy' in q.keys():
            ports += [q.channels['xy_I'],q.channels['xy_Q']]
        if 'z' in q.keys():
            ports += [q.channels['z']]
    return ports
    

def set_microwaveSource(deviceList,freqList,powerList):
    """set frequency and power for microwaveSource devices
    """
    for i in range(len(deviceList)):
        deviceList[i].set_freq(freqList[i])
        deviceList[i].set_power(powerList[i])
    return
        


def makeSequence_readout(waveServer,qubits,FS):
    """
    waveServer: zilabrad.instrument.waveforms
    This version only, consider one zurich_qa device
    FS: sampling rates
    """
    wave_readout_func = [waveforms.NOTHING,waveforms.NOTHING]
    for q in qubits:
        if 'do_readout' in q.keys():
            if 'r' in q.keys(): 
                wave_readout_func[0] += q.r[0]
                wave_readout_func[1] += q.r[1]
            else:
                print('Error! No readout pulse!')
    
    ## this is just set parameters, not really generate a list, which is slow
    q_ref = qubits[0]
    start = 0.
    end = q_ref['readout_len']['s']
    waveServer.set_tlist(start=start,end=end,fs=FS)
    wave_readout = [waveServer.func2array(wave_readout_func[i],start,end) for i in [0,1]]
    return wave_readout


def makeSequence_AWG(waveServer,qubits,FS):
    """
    waveServer: zilabrad.instrument.waveforms
    FS: sampling rates
    """
    wave_AWG = []

    ## This version consider all of the ports of AWG require the same sampling points


    ### ----bias_start previously -----  XY,Z, pulse quantum getes... ------ readout ----- bias_end ---
    q_ref = qubits[0]
    start = -Unit2SI(q_ref['bias_start'])
    end = Unit2SI(q_ref['bias_end']) + Unit2SI(q_ref['experiment_length'])
    
    ## this is just set parameters, not really generate a list, which is slow
    waveServer.set_tlist(start,end,fs=FS)
    
    for q in qubits: 
        ## line [DC]
        if 'dc' in q.keys():
            wave_AWG += [waveServer.func2array(q.dc,start,end,FS)]
            
        ## line [xy] I,Q
        if 'xy' in q.keys():
            wave_AWG += [waveServer.func2array((q.xy)[i],start,end,FS) for i in [0,1]]
            
        ## line [z]
        if 'z' in q.keys():
            wave_AWG += [waveServer.func2array(q.z,start,end,FS)]
            
    print(np.asarray(wave_AWG))
    return wave_AWG


def runQubits(qubits,exp_devices):
    """ generally for running multiqubits

    Args:
        qubits (list): a list of dictionary
        _runQ_servers (list/tuple): instances used to control device
    
    TODO:
        (1) check _is_runfirst=True? Need run '_mpAwg_init' at first running;
        (2) clear q.xy/z/dc and their array after send();
    """
    qa,hd,mw,mw_r,waveServer = exp_devices[:5]
    
    
    wave_AWG = makeSequence_AWG(waveServer,qubits,hd.FS)
    wave_readout = makeSequence_readout(waveServer,qubits,qa.FS)
    
    q_ref = qubits[0]
    if hd.pulse_length_s != q_ref['experiment_length']:
        hd.pulse_length_s = q_ref['experiment_length']
        qa.set_adc_trig_delay(q_ref['bias_start'][s]+hd.pulse_length_s)
    
    
    
    ## Now it is only two microwave sources, in the future, it should be modified
    set_microwaveSource(deviceList = [mw,mw_r],
                        freqList = [q_ref['xy_mw_fc'],q_ref['readout_mw_fc']],
                        powerList = [q_ref['xy_mw_power'],q_ref['readout_mw_power']])


    # initialization
    if 'do_init' not in qubits[0].keys():
        _mpAwg_init(qubits,exp_devices[:4])
        qubits[0]['do_init']=True
        logging.info('do_init')
    
    qubit_ports = getQubits_awgPort(qubits)
    hd.send_waveform(waveform=wave_AWG, ports=qubit_ports)
    qa.send_waveform(waveform=wave_readout)

    ## start to run experiment
    hd.awg_open()
    qa.awg_open()

    data = qa.get_data()
    return data

