# -*- coding: utf-8 -*-
"""
Created on 2020.09.09 20:57
Controller for Zurich Instruments
@author: , Huang Wenhui
"""


import zhinst.utils ## create API object
import textwrap ## to write sequencer's code
import time ## show total time in experiments
import matplotlib.pyplot as plt ## give picture
import numpy as np 
from numpy import pi
from math import ceil
import itertools

# from zurich_qa import zurich_qa ## quantum analyzer class
# from zurich_hd import zurich_hd ## hdawg instruments class
# from microwave_source import microwave_source ## control microwave source by visa
import pyvisa
from importlib import reload
from zurichHelper import *
import zurichHelper


"""
import for pylabrad
"""
from pyle import sweeps
import labrad
import time
import numpy as np
from pyle.util import sweeptools as st
from pyle.workflow import switchSession
from pyle.pipeline import returnValue, FutureList
from labrad.units import Unit,Value

_unitSpace = ('V', 'mV', 'us', 'ns','s', 'GHz', 'MHz','kHz','Hz', 'dBm', 'rad','None')
V, mV, us, ns,s, GHz, MHz,kHz,Hz, dBm, rad,_l  = [Unit(s) for s in _unitSpace]

ar = st.r


cxn=labrad.connect()
dv = cxn.data_vault
# specify the sample, in registry   
ss = switchSession(cxn,user='Ziyu',session=None) 

    
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



def power2amp(power):
    """ 
    convert 'dBm' to 'V' for 50 Ohm (only value)
    Based on --> 0.5*Vpp=sqrt(2)*sqrt(50e-3*10^(dBm/10)); 
    """
    return 10**(power/20-0.5)

"""
pylabrad End
"""





def Unit2SI(a):
    if type(a) is not Value:
        return a
    elif a.unit in [GHz,MHz]:
        return a['Hz']
    elif a.unit in [ns,us]:
        return a['s']
    else:
        return a[a.unit] 

def Unit2num(a):
    if type(a) is not Value:
        return a
    else:
        return a[a.unit] 

def s21_scan(sample,measure=0,stats=100,freq=ar[6.4:6.9:0.002,GHz],
    bias=0.5*V,power=-10*dBm,phase=0.0,name='s21_scan',des='',back=True,noisy=True):
    """ 
        sample: select experimental parameter from registry;
        stats: Number of Samples for one sweep point;
    """
    _t0_ = time.time()
    ## load parameters 
    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    q = qubits[measure]


    ## set some parameters name;
    axes = [(freq,'freq'),(bias,'bias'),(power,'power'),(phase,'phase')]
    deps = [('Amplitude','s21 for','?'),('Phase','s21 for','rad'),
                ('I','',''),('Q','','')]

    kw = {'stats': stats}

    # create dataset
    dataset = sweeps.prepDataset(sample, name+des, axes, deps,measure=measure, kw=kw)
    dataset_create(dv,dataset)
    indeps = dataset.independents
    print(indeps)

    w_qa,w_hd = mpAwg_init(q,stats)
    # qa.daq.setDouble('/{:s}/sigins/*/range'.format(qa.id), 0.1) # 输入量程0.1V

    def runQ(para_list):
        freq,bias,power,phase = para_list
        power = power2amp(power) ## consider 0.5*Vpp for 50 Ohm impedance

        ## set device parameter
        # q['readout_fc'] = (freq + sb_freq)*Hz
        sb_freq = q['readout_fc'][Hz]-freq
        # mw.set_freq(q['readout_fc'][Hz])
        # time.sleep(0.1)
        qa.set_qubit_frequency([sb_freq])
        # qa.daq.setDouble('/{:s}/oscs/0/freq'.format(qa.id),sb_freq) ## set internal oscs freq


        ## write waveforms 
        q_dc = w_hd.square(amp=bias,start=-q['bias_start'][s],end=hd.pulse_length_s+qa.pulse_length_s+q['bias_end'][s])
        start = 0
        ## 如果有xy/z pulse,就在这加,然后更改start;
        hd.pulse_length_s = start ## 记录hd部分涉及的pulse length;

        ## 结束hd脉冲,开始设置读取部分
        q_r = [w_qa.cosine(amp=power,freq=sb_freq,phase=phase,start=start,end=start+q['readout_len'][s]),
               w_qa.sine(amp=power,freq=sb_freq,phase=phase,start=start,end=start+q['readout_len'][s])]
        # q_r = [w_qa.square(amp=power,start=0,end=0+q['readout_len'][s]),
        #        w_qa.square(amp=power,start=0,end=0+q['readout_len'][s])]

        ## reload new waveform in this runQ
        hd.reload_waveform(waveform=q_dc)
        qa.reload_waveform(waveform=q_r)

        ## start to run experiment
        hd.awg_open()
        qa.awg_open()
        data = qa.get_data()

        ## analyze data and return
        for _d_ in data:
            amp = np.mean(np.abs(_d_))/power ##20*np.log10(np.mean(np.abs(_d_))/power) ## unit: dB; only relative strength;
            phase = np.mean(np.angle(_d_))
            Iv = np.mean(np.real(_d_))
            Qv = np.mean(np.imag(_d_))
        ## multiply channel should unfold to a list for return result
        result = [amp,phase,Iv,Qv]
        return result 

        


    result_list = []
    ## start to running ##
    for axes_scan in gridSweep(axes):
        _para_ = [Unit2SI(a) for a in axes_scan[0]]
        indep = [Unit2num(a) for a in axes_scan[1]]

        result = runQ(_para_)
        
        data_send = indep + result
        if noisy:
            print(', '.join([format(x,'.3f') for x in data_send])) ## show in scientific notation
        result_list.append(data_send)
        dv.add(data_send.copy()) ## save value to dataVault


    stop_device() ## stop all device running
    print('%r use time:%r'%(name+des,time.time()-_t0_))
    if back:
        return result_list



if __name__ == '__main__':
    modes=[1,2]
    bringup_device(modes=modes)

 

