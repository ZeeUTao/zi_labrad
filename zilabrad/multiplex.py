# -*- coding: utf-8 -*-
"""
Controller for Zurich Instruments

The daily experiments are implemented via various
function (s21_scan, ramsey...)

Created on 2020.09.09 20:57
@author: Huang Wenhui, Tao Ziyu
"""

import gc
import logging  # python standard module for logging facility
import time  # show total time in experiments
import matplotlib.pyplot as plt  # give picture
from functools import wraps, reduce
import functools
import numpy as np
from numpy import pi
import itertools

from zilabrad.instrument.QubitContext import qubitContext

from zilabrad.qubitServer import RunAllExperiment,gridSweep,runQubits,reset_qubits,runQubits_raw

from zilabrad import pulse

from zilabrad.plots import dataProcess

from zilabrad.pyle import sweeps
from zilabrad.pyle.util import sweeptools
from zilabrad.pyle.registry import AttrDict
from zilabrad.pyle.envelopes import Envelope, NOTHING
from zilabrad.pyle.tools import Unit2SI,convertUnits


from labrad.units import Unit, Value
_unitSpace = ('V', 'mV', 'us', 'ns', 's', 'GHz',
              'MHz', 'kHz', 'Hz', 'dBm', 'rad', 'None')
V, mV, us, ns, s, GHz, MHz, kHz, Hz, dBm, rad, _l = [
    Unit(s) for s in _unitSpace]
ar = sweeptools.RangeCreator()






##############################
##  preprocessing register  ##
##############################
def loadQubits(Sample,measure=None):
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
    sample = Sample.copy()
    qubits = AttrDict({name:sample['Qubits'][name] for name in sample['Qubits']['config']})
    ## convert measure to string-name list
    if isinstance(measure,int):
        measure = [measure]
    if isinstance(measure,list):
        for k in range(len(measure)):
            if isinstance(measure[k],str):
                if measure[k] not in sample['Qubits']['config']:
                    raise Exception('[%s] not in Qubits.config'%measure[k])
            else:
                measure[k] = sample['Qubits']['config'][int(measure[k])]
    if measure is None:
        measure = sample['Qubits']['config']
    if isinstance(measure,str):
        measure = [measure]

    del sample['Qubits']
    ## convert to AttrDict for gate & correct key
    for qb in qubits.values():
        for key in qb['gate']:
            qb['gate'][key] = AttrDict(qb['gate'][key])
        for key in qb['correct']:
            qb['correct'][key] = AttrDict(qb['correct'][key])

    ## load Devices Settings in sample._taskflow
    for dev_name in sample['run']:
        sample['_taskflow'][dev_name] = sample[dev_name].copy()

    
    ## basic qubits parameter setting
    for qb in qubits.values(): 
        # set qubit demod freq for each qubit
        qb['demod_freq'] = (qb['readout_freq']-LO_ctrl(sample,qb,'frequency',TYPE='readout'))
        if abs(Unit2SI(qb['demod_freq'])) > 900e6:
            raise Exception('qubit.demod_freq out of range, please check mw_r.frequency!')

    return sample, qubits, measure




############################
## Experimental decorator ##
############################
def expfunc_decorator(func):
    """
    can define some thing before or after 'func'
    allow interrupt func running by 'ctrl+C'
    Before:
        use gc.collect() to release memory
    After:
        stop all AWG running & MW output, 
        print current time and total time 
        of Experimental function
    """
    def stop_device():
        """  close all device;
        """
        qContext = qubitContext()
        awg_devices = qContext.get_servers_group(
            name='ArbitraryWaveGenerator').values()
        for dev in awg_devices:
            dev.awg_run(_run=False)

        qa_devices = qContext.get_servers_group(
            name='QuantumAnalyzer').values()
        for dev in qa_devices:
            dev.awg_run(_run=False)

        mw_devices = qContext.get_servers_group(
            name='MicrowaveSource').values()
        for dev in mw_devices:
            dev.output(False)
        return

    @wraps(func)
    def wrapper(*args, **kwargs):
        ## use garbage collector to release memory 
        gc_var = gc.collect()
        print(f"garbage collect {gc_var}")

        start_ts = time.time()
        try:
            result = func(*args, **kwargs)
        except KeyboardInterrupt:
            # interrupt this function by 'ctrl+C'
            print('KeyboardInterrupt')
        # stop all device output
        stop_device()  
        print(time.strftime("%Y-%m-%d %X", time.localtime()))
        print('Use time: %.1f s'%(time.time()-start_ts))
        return result
    return wrapper







############################
##      convert unit      ##
############################
@convertUnits(power='dBm')
def power2amp(power):
    """
    convert 'dBm' to 'V' for 50 Ohm (only value)
    Based on --> 0.5*Vpp=sqrt(2)*sqrt(50e-3*10^(dBm/10));
    """
    return 10**(power/20-0.5)

@convertUnits(amp='V')
def amp2power(amp):
    """
    convert 'dBm' to 'V' for 50 Ohm (only value)
    Based on --> 0.5*Vpp=sqrt(2)*sqrt(50e-3*10^(dBm/10));
    """
    return 20*np.log10(amp)+10






############################
## control Settings tools ##
############################
def LO_ctrl(sample,qb,command,parameter=None,TYPE='xy'):
    dev_name = qb['channels']['LO_%s'%TYPE]
    if isinstance(parameter,type(None)):
        return sample[dev_name][command]
    else:
        sample[dev_name][command] = parameter

def set_stats(sample,qb,stats):
    dev_name = qb['channels']['readout'].split('-')[0]
    sample[dev_name]['set_result_sample']=stats






############################
## control sequence tools ##
############################
@convertUnits(dt='s')
def moving(qubits,dt):
    for qb in qubits.values():
        qb['_start'] += dt


def add_qubitsDC(qubits, sample):
    bias_start = -sample['waveGenerator']['bias_rise']
    bias_end = sample['waveGenerator']['awg_pulse_length']\
                + sample['waveGenerator']['bias_fall']
    for qb in qubits.values():
        bias = qb['bias']
        if type(bias) not in [float, int]:
            bias = bias[bias.unit]
        if abs(bias)<1e-5: 
            continue ## bias=0 -> pass this qubit
        if 'z' not in qb:
            qb['z'] = NOTHING
        qb['z'] += pulse.rect(amp=bias,
            start=bias_start,end=bias_end)
    return


def add_readoutPulse(q,start,sample=None,SyncDemod=True):
    if 'r' not in q:
        q['r'] = NOTHING
    if hasattr(q['readout_amp'], 'unit'):
        if q['readout_amp'].unit in ['dBm']:
            q['readout_amp'] = power2amp(q['readout_amp'])
    q['r'] += pulse.readout(
        amp=q['readout_amp'], phase=0, 
        freq=q['demod_freq'], 
        start=start, length=q['readout_len'])
    if SyncDemod:
        q['demod_start'] = start




def add_spectroscopyPulse(q,amp,start,length,freq=0,phase=0):
    if 'xy' not in q:
        q['xy'] = NOTHING
    q['xy']+=pulse.spectroscopy(amp=amp,start=start,freq=freq,length=length)
    


def add_zpaPulse(q,amp,start,length):
    if 'z' not in q:
        q['z'] = NOTHING
    q['z']+=pulse.rect(start=start,amp=amp,length=length)





############################
##      Gate function     ##
############################

def add_XYgate(sample,q,start,factor=1,phi=0,amp=None,length=None,alpha=None,df=0):
    ## need be rewritten
    ## no drag 
    if 'xy' not in q: 
        q['xy'] = NOTHING
    _gate = dict(q['gate']['pipulse'])
    if amp is None: 
        amp = factor*_gate['Amp']
    if length is None: 
        length = _gate['Len']
    if alpha is None:
        alpha = _gate['Alpha']

    sb_freq = Unit2SI(q['f10']-LO_ctrl(sample,q,'frequency',TYPE='xy'))+df
    # q['xy']+=pulse.spectroscopy(amp=amp,start=start,freq=sb_freq,length=length,phase=phi)
    q['xy'] += pulse.gauss_gate(amp=amp,start=start,freq=sb_freq,length=length,phase=phi,alpha=alpha)
    return



def add_halfXYgate(sample,q,start,factor=1,phi=0,amp=None,length=None,df=0):
    ## need be rewritten
    ## no drag 
    if 'xy' not in q: 
        q['xy'] = NOTHING
    _gate = dict(q['gate']['piHalfpulse'])
    if amp is None: 
        amp = factor*_gate['Amp']
    if length is None: 
        length = _gate['Len']
    if alpha is None:
        alpha = _gate['Alpha']

    sb_freq = Unit2SI(q['f10']-LO_ctrl(sample,q,'frequency',TYPE='xy'))+df
    # q['xy']+=pulse.spectroscopy(amp=amp,start=start,freq=sb_freq,length=length,phase=phi)
    q['xy'] += pulse.gauss_gate(amp=amp,start=start,freq=sb_freq,length=length,phase=phi,alpha=alpha)
    return





############################
## Variable & Result Tool ##
############################
def MultiChannel_Deps(measure_qubits,raw=True,prob=False,level=2):
    deps = []
    for qb_name in measure_qubits:
        name = str(qb_name)
        if raw:
            deps += [('Amplitude', 's21 for %s'%name, 'a.u.'),
                     ('Phase', 's21 for %s'%name, 'rad'),
                     ('I', '%s'%name, ''), ('Q', '%s'%name, '')]
        if prob:
            if level == 2:
                deps += [('prob |1>', '%s'%name, '')]
            else:
                for k in range(level):
                    deps += [('prob |%d>'%k, '%s'%name, '')]
    return deps


def MultiChannel_Result(qubits,data_dict,raw=True,prob=False,level=2):
    result = []
    for qb_name in data_dict.keys():
        qb = qubits[qb_name]
        _d_ = data_dict[qb_name]
        if raw:
            if hasattr(qb['readout_amp'], 'unit'):
                if qb['readout_amp'].unit in ['dBm']:
                    qb['readout_amp'] = power2amp(qb['readout_amp'])
            result += [np.abs(np.mean(_d_))/Unit2SI(qb['readout_amp'])] ## amp
            result += [np.angle(np.mean(_d_))] ## phase
            result += [np.real(np.mean(_d_))] ## Iv
            result += [np.imag(np.mean(_d_))] ## Qv
        if prob:
            ## need rewrite!
            p_val = tunneling([qb], [_d_], level=level)
            if level == 2:
                result += [p_val[1]]
            else:
                result += list(p_val)
    return result






############################
##  Result Analysis Tool  ##
############################

def tunneling(qubits, data, level=2):
    """ get probability for 1,2,3...N qubits with level (2,3,4,....)
    Args:
        qubits (dict): qubit information in registry
        data (list): list of IQ data (array of complex number) for N qubits
        level (int): level of qubit
    """
    qNum = len(qubits)
    counts_num = len(data[0])
    binary_count = np.zeros((counts_num), dtype=float)

    def get_meas(data0, q, Nq=level):
        # if measure 1 then return 1
        sigs = data0

        total = len(sigs)
        distance = np.zeros((total, Nq))
        for i in np.arange(Nq):
            center_i = q['IQcenter'][i][0] + \
                1j*q['IQcenter'][i][1]
            distance_i = np.abs(sigs - center_i)
            distance[:, i] = distance_i
        tunnels = np.zeros((total,))
        for i in np.arange(total):
            distancei = distance[i]
            tunneli = np.int(np.where(distancei == np.min(distancei))[0])
            tunnels[i] = tunneli
        return tunnels

    for i in np.arange(qNum):
        binary_count += get_meas(data[i], qubits[i]) * (level**(qNum-1-i))

    res_store = np.zeros((level**qNum))
    for i in np.arange(level**qNum):
        res_store[i] = np.sum(binary_count == i)

    prob = res_store/counts_num
    return prob













##############################
###        Test tool       ###
##############################
from IPython.display import clear_output
def live_plot(q):
    tlist = np.arange(0,10e-6,0.5e-9)
    xy_array = q.xy(tlist)
    z_array = q.z(tlist)
    r_array = q.r(tlist)
    # clear_output(wait=True)
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(1,1,1)
    plt.plot(tlist,xy_array.real)
    plt.plot(tlist,xy_array.imag)
    plt.plot(tlist,z_array.real-1)
    plt.plot(tlist,r_array.real-2)
    plt.plot(tlist,r_array.imag-2)
    plt.yticks([0,-1,-2],['xy','z','r'])
    plt.show()
    time.sleep(0.2)

    


##############################
### main sweeping function ###
##############################


####################
### single qubit ###
####################
@expfunc_decorator
def s21_scan(Sample, measure=0, freq=6.0*GHz, mw_power=None, bias=None, power=None,
             name='s21_scan', des=''):
    """
    s21 scanning:
        measure -> controlled & read qubit index
        freq[Hz] -> readin pulse frequency after mixer
        bias[None,V] -> DC signal in Z-line during running
        power[dBm] -> AWG amplitude in Read-line, will convert [dBm] to [V]
        mw_power[dBm] -> Microwave Source(read) power
    """
    sample, qubits, Qubits = loadQubits(Sample, write_access=True)
    q = qubits[measure]

    for qb in qubits:
        qb['qa_demod_freq'] = (qb['readout_freq']-LO_ctrl(Sample,qb,'frequency',TYPE='readout'))

    if freq is None:
        freq = q['readout_freq']
    if bias is None:
        bias = q['bias']
    if power is None:
        power = q['readout_amp']
    if mw_power is None:
        mw_power = LO_ctrl(Sample,q,'power',TYPE='readout')

    # set some parameters name;
    axes = [(freq, 'freq'), (bias, 'bias'), (power, 'power'),
            (mw_power, 'mw_power')]
    deps = MultiChannel_Deps([q],raw=True,prob=False)
    kw = {}

    # create dataset
    dataset = sweeps.prepDataset(
        sample, name+' '+des, axes, deps, measure=measure, kw=kw)

    def runSweeper(devices, para_list):
        freq, bias, power, mw_power = para_list
        clear_qubitsChannels([q])
        q['bias'] = bias
        q['readout_amp'] = power*dBm
        LO_ctrl(sample,q,'frequency',
            (freq - Unit2SI(q['qa_demod_freq']))*Hz,TYPE='readout')
        LO_ctrl(sample,q,'power',
            mw_power*dBm,TYPE='readout')

        ## start this sequence
        start = 0
        ## set readout pulse & demod window
        for qb in qubits:
            qb['do_readout'] = True
            add_readoutPulse(qb,start,sample,SyncDemod=True)
        ## set bias for qubits
        add_qubitsDC(qubits,sample)
        # get result value
        data = runQubits([q],sample['Settings'])

        result = MultiChannel_Result([q],data,raw=True,prob=False)
        return result

    axes_scans = gridSweep(axes)
    result_list = RunAllExperiment(runSweeper, axes_scans, dataset)






@expfunc_decorator
def s21_dispersive(Sample, measure=0, freq=6.0*GHz, 
                   mw_power=None, bias=None, power=None,
                   name='s21_dispersive', des=''):
    """
    s21 dispersive:
        measure -> controlled & read qubit index
        freq[Hz] -> readin pulse frequency after mixer
        bias[None,V] -> DC signal in Z-line during running
        power[dBm] -> AWG amplitude in Read-line, will convert [dBm] to [V]
        mw_power[dBm] -> Microwave Source(read) power
    """
    sample, qubits, Qubits = loadQubits(Sample, write_access=True)
    q = qubits[measure]

    for qb in qubits:
        qb['qa_demod_freq'] = (qb['readout_freq']-LO_ctrl(Sample,qb,'frequency',TYPE='readout'))

    if freq is None:
        freq = q['readout_freq']
    if bias is None:
        bias = q['bias']
    if power is None:
        power = q['readout_amp']
    if mw_power is None:
        mw_power = LO_ctrl(Sample,q,'power',TYPE='readout')

    # set some parameters name;
    axes = [(freq, 'freq'), (bias, 'bias'), (power, 'power'),
            (mw_power, 'mw_power')]
    deps = [('Amplitude|0>', 's21 for %s' % q.__name__,''),
            ('Phase|0>', 's21 for %s' % q.__name__,''),
            ('I|0>', '', ''),('Q|0>', '', ''),
            ('Amplitude|1>', 's21 for %s' % q.__name__,''),
            ('Phase|1>', 's21 for %s' % q.__name__,''),
            ('I|1>', '', ''),('Q|1>', '', ''),
            ('SNR', '', '')]
    kw = {}

    # create dataset
    dataset = sweeps.prepDataset(
        sample, name+' '+des, axes, deps, measure=measure, kw=kw)

    def runSweeper(devices, para_list):
        freq, bias, power, mw_power = para_list
        
        q['bias'] = bias
        q['readout_amp'] = power*dBm
        LO_ctrl(sample,q,'frequency',
            (freq - Unit2SI(q['qa_demod_freq']))*Hz,TYPE='readout')
        LO_ctrl(sample,q,'power',
            mw_power*dBm,TYPE='readout')

        ## start this sequence: |0>
        clear_qubitsChannels([q])
        start = 0
        add_XYgate(Sample,q,start,factor=1)
        start += dict(q['gate']['pipulse'])['Len']['s']
        ## set readout pulse & demod window
        for qb in qubits:
            qb['do_readout'] = True
            add_readoutPulse(qb,start,sample,SyncDemod=True)
        ## set bias for qubits
        add_qubitsDC(qubits,sample)
        # get result value
        data0 = runQubits([q],sample['Settings'])


        ## start this sequence: |1>
        clear_qubitsChannels([q])
        start = 0
        add_XYgate(Sample,q,start,factor=1)
        start += dict(q['gate']['pipulse'])['Len']['s']
        ## set readout pulse & demod window
        for qb in qubits:
            qb['do_readout'] = True
            add_readoutPulse(qb,start,sample,SyncDemod=True)
        ## set bias for qubits
        add_qubitsDC(qubits,sample)
        # get result value
        data1 = runQubits([q],sample['Settings'])



        result = MultiChannel_Result([q],data0,raw=True,prob=False)
        result+= MultiChannel_Result([q],data1,raw=True,prob=False)
        result+= [np.abs((result[2]-result[6])+1j*(result[3]-result[7]))]
        return result

    axes_scans = gridSweep(axes)
    result_list = RunAllExperiment(runSweeper, axes_scans, dataset)





@expfunc_decorator
def spectroscopy(Sample, measure=0, freq=6.0*GHz, sb_freq=0*MHz,
                 specLen=1*us, specAmp=0.05, bias=None, zpa=0,
                 name='spectroscopy', des=''):
    """ 
    """
    sample, qubits, Qubits = loadQubits(Sample, write_access=True)
    q = qubits[measure]

    for qb in qubits:
        qb['qa_demod_freq'] = (qb['readout_freq']-LO_ctrl(Sample,qb,'frequency',TYPE='readout'))

    if bias is None:
        bias = q['bias']

    # set some parameters name;
    axes = [(freq, 'freq'), (sb_freq, 'sb_freq'), 
            (specLen, 'specLen'), (specAmp, 'specAmp'),
            (bias, 'bias'), (zpa, 'zpa')]
    deps = MultiChannel_Deps([q],raw=True,prob=True)
    kw = {}

    # create dataset
    dataset = sweeps.prepDataset(
        sample, name+' '+des, axes, deps, measure=measure, kw=kw)

    def runSweeper(devices, para_list):
        freq, sb_freq, specLen, specAmp, bias, zpa = para_list
        clear_qubitsChannels([q])
        q['bias'] = bias
        LO_ctrl(sample,q,'frequency',(freq-sb_freq)*Hz,TYPE='xy')

        ## start this sequence
        start = 0
        add_spectroscopyPulse(q,specAmp,start,specLen,freq=sb_freq)
        add_zpaPulse(q,zpa,start,specLen)
        start += specLen

        ## set readout pulse & demod window
        for qb in qubits:
            qb['do_readout'] = True
            add_readoutPulse(qb,start,sample,SyncDemod=True)
        ## set bias for qubits
        add_qubitsDC(qubits,sample)


        # live_plot(q)
        # result=[-1]*5
        # # get result value
        data = runQubits([q],sample['Settings'])

        result = MultiChannel_Result([q],data,raw=True,prob=True)
        return result

    axes_scans = gridSweep(axes)
    result_list = RunAllExperiment(runSweeper, axes_scans, dataset)







@expfunc_decorator
def rabi(Sample, measure=[0], piamp=0.1, pilen=40*ns, delay=0*ns, reps = 1000,
         df=0*MHz, bias=None, zpa=0, name='rabi', des=''):
    """
    """
    devices, qubits, measure = loadQubits(Sample,measure)
    q = qubits[measure[0]]

    # set some parameters name;
    axes = [(piamp, 'piamp'), (pilen, 'pilen'), (delay,'delay'),
            (df, 'df'), (bias, 'bias'), (zpa, 'zpa')]
    deps = MultiChannel_Deps(measure,raw=True,prob=True)
    kw = {}

    # create dataset
    dataset = sweeps.prepDataset(
        Sample, name+' '+des, axes, deps, measure=measure, kw=kw)

    def runSweeper(piamp,pilen,delay,df,bias,zpa):
        if bias is not None:
            q['bias'] = bias

        ## start this sequence
        start = delay
        for k in range(reps):
            add_XYgate(devices,q,start,amp=piamp,length=pilen,df=df)
            add_zpaPulse(q,zpa,start,pilen)
            start += pilen+10e-9
        ## set readout pulse & demod window
        for qb in qubits.values():
            qb['do_readout'] = True
            add_readoutPulse(qb,start,devices,SyncDemod=True)
        ## set bias for qubits
        add_qubitsDC(qubits,devices)

        ## get result
        data = runQubits(qubits,devices,measure)
        # ts = np.arange(0,10e-6,1/2.4e9)
        # for qb in qubits.values():
        #     if 'xy' in qb:
        #         qb['xy'] = qb['xy'](ts)[:int((pilen+20e-9)*2.4e9*num)]
        #     if 'z' in qb:
        #         qb['z'] = qb['z'](ts)[:int((pilen+20e-9)*2.4e9*num)]
        # data = runQubits_raw(qubits,devices,measure)
        result = MultiChannel_Result(qubits,data,raw=True,prob=True)
        return result

    axes_scans = gridSweep(axes)
    result_list = RunAllExperiment(runSweeper, axes_scans, dataset)
    return q





@expfunc_decorator
def IQraw(Sample,measure=0,stats=None,reps=1,back=False,name='IQraw',des=''):
    """ 
    """
    sample, qubits, Qubits = loadQubits(Sample, write_access=True)
    q = qubits[measure]

    for qb in qubits:
        qb['qa_demod_freq'] = (qb['readout_freq']-LO_ctrl(Sample,qb,'frequency',TYPE='readout'))

    if stats is not None:
        set_stats(sample,q,stats)

    # set some parameters name;
    axes = [] #[(reps, 'reps')]
    deps = [('Is', '|0>', ''), ('Qs', '|0>', ''),
            ('Is', '|1>', ''), ('Qs', '|1>', '')]
    kw = {}

    # create dataset
    dataset = sweeps.prepDataset(
        sample, name+' '+des, axes, deps, measure=measure, kw=kw)

    def runSweeper(devices, para_list):
        # reps = para_list[0]

        ## start this sequence: |0>
        clear_qubitsChannels([q])
        start = 0
        add_XYgate(Sample,q,start,factor=0)
        start += dict(q['gate']['pipulse'])['Len']['s']
        ## set readout pulse & demod window
        for qb in qubits:
            qb['do_readout'] = True
            add_readoutPulse(qb,start,sample,SyncDemod=True)
        ## set bias for qubits
        add_qubitsDC(qubits,sample)
        # get result value
        data0 = runQubits([q],sample['Settings'])


        ## start this sequence: |1>
        clear_qubitsChannels([q])
        start = 0
        add_XYgate(Sample,q,start,factor=1)
        start += dict(q['gate']['pipulse'])['Len']['s']
        ## set readout pulse & demod window
        for qb in qubits:
            qb['do_readout'] = True
            add_readoutPulse(qb,start,sample,SyncDemod=True)
        ## set bias for qubits
        add_qubitsDC(qubits,sample)
        # get result value
        data1 = runQubits([q],sample['Settings'])


        # date process
        Is0 = np.real(data0[measure])
        Qs0 = np.imag(data0[measure])
        Is1 = np.real(data1[measure])
        Qs1 = np.imag(data1[measure])

        result = [Is0, Qs0, Is1, Qs1]
        return result


    axes_scans = gridSweep(axes)
    result_list = RunAllExperiment(runSweeper, axes_scans, dataset, raw=True)
    if back:
        return result_list[0]






@expfunc_decorator
def T1(Sample,measure=0,delay=10*us,bias=None,zpa=0,name='T1',des=''):
    """
    """
    sample, qubits, Qubits = loadQubits(Sample, write_access=True)
    q = qubits[measure]

    for qb in qubits:
        qb['qa_demod_freq'] = (qb['readout_freq']-LO_ctrl(Sample,qb,'frequency',TYPE='readout'))

    if bias is None:
        bias = q['bias']

    # set some parameters name;
    axes = [(delay, 'delay'),(bias, 'bias'), (zpa, 'zpa')]
    deps = MultiChannel_Deps([q],raw=True,prob=True)
    kw = {}

    # create dataset
    dataset = sweeps.prepDataset(
        sample, name+' '+des, axes, deps, measure=measure, kw=kw)

    def runSweeper(devices, para_list):
        delay,bias,zpa = para_list
        clear_qubitsChannels([q])
        q['bias'] = bias

        ## start this sequence
        start = 0
        add_XYgate(Sample,q,start)
        start += dict(q['gate']['pipulse'])['Len']['s']
        add_zpaPulse(q,zpa,start,delay)
        start += delay

        ## set readout pulse & demod window
        for qb in qubits:
            qb['do_readout'] = True
            add_readoutPulse(qb,start,sample,SyncDemod=True)
        ## set bias for qubits
        add_qubitsDC(qubits,sample)

        # get result value
        data = runQubits([q],sample['Settings'])

        result = MultiChannel_Result([q],data,raw=True,prob=True)
        return result

    axes_scans = gridSweep(axes)
    result_list = RunAllExperiment(runSweeper, axes_scans, dataset)







@expfunc_decorator
def ramsey(Sample,measure=0,delay=10*us,fringeFreq=10*MHz, 
           PHASE=np.pi/2,bias=None,zpa=0,name='ramsey',des=''):
    """
    """
    sample, qubits, Qubits = loadQubits(Sample, write_access=True)
    q = qubits[measure]

    for qb in qubits:
        qb['qa_demod_freq'] = (qb['readout_freq']-LO_ctrl(Sample,qb,'frequency',TYPE='readout'))

    if bias is None:
        bias = q['bias']

    # set some parameters name;
    axes = [(delay, 'delay'),(fringeFreq,'fringeFreq'),
            (PHASE,'phase'),(bias, 'bias'), (zpa, 'zpa')]
    deps = MultiChannel_Deps([q],raw=True,prob=True)
    kw = {}

    # create dataset
    dataset = sweeps.prepDataset(
        sample, name+' '+des, axes, deps, measure=measure, kw=kw)

    def runSweeper(devices, para_list):
        delay,fringeFreq,PHASE,bias,zpa = para_list
        clear_qubitsChannels([q])
        q['bias'] = bias

        ## start this sequence
        start = 0
        add_XYgate(Sample,q,start,factor=0.5)
        start += dict(q['gate']['pipulse'])['Len']['s']
        add_zpaPulse(q,zpa,start,delay)
        start += delay
        add_XYgate(Sample,q,start,factor=0.5,phi=PHASE+fringeFreq*delay*2*np.pi)
        start += dict(q['gate']['pipulse'])['Len']['s']
        ## set readout pulse & demod window
        for qb in qubits:
            qb['do_readout'] = True
            add_readoutPulse(qb,start,sample,SyncDemod=True)
        ## set bias for qubits
        add_qubitsDC(qubits,sample)

        live_plot(q)
        # get result value
        data = runQubits([q],sample['Settings'])

        result = MultiChannel_Result([q],data,raw=True,prob=True)
        return result

    axes_scans = gridSweep(axes)
    result_list = RunAllExperiment(runSweeper, axes_scans, dataset)





####################
### multi-qubits ###
####################
@expfunc_decorator
def s21_multi(Sample,ctrl=['q1'],mw_df=6.0*GHz, power=-30*dBm, bias=0,
              name='s21_multi', des=''):
    """
    s21 scanning:
        sweep mw_df nearly mw frequency, 
        same power & bias in each qubit,
        readout qubits together
    """
    sample, qubits, Qubits = loadQubits(Sample, write_access=True)

    fc_copy = sample['anritsu_r_1']['frequency']
    for qb in qubits:
        qb['qa_demod_freq'] = (qb['readout_freq']-fc_copy)

    # set some parameters name;
    axes = [(mw_df, 'mw_df'),(bias, 'bias'),(power, 'power')]
    deps = MultiChannel_Deps(qubits,raw=True,prob=False)
    kw = {}
    # create dataset
    dataset = sweeps.prepDataset(
        sample, name+' '+des, axes, deps, measure=list(range(len(qubits))), kw=kw)


    def runSweeper(devices, para_list):
        df, bias, power = para_list
        clear_qubitsChannels(qubits)
        for qb in qubits:
            qb['readout_amp'] = power*dBm
        for qname in ctrl:
            sample[qname]['bias'] = bias
        sample['Settings']['anritsu_r_1']['frequency'] = fc_copy + df*Hz

        ## start this sequence
        start = 0
        ## set readout pulse & demod window
        for qb in qubits:
            qb['do_readout'] = True
            add_readoutPulse(qb,start,sample,SyncDemod=True)
        ## set bias for qubits
        add_qubitsDC(qubits,sample)
        # get result value
        data = runQubits(qubits,sample['Settings'])

        result = MultiChannel_Result(qubits,data,raw=True,prob=False)
        return result

    axes_scans = gridSweep(axes)
    result_list = RunAllExperiment(runSweeper, axes_scans, dataset)








@expfunc_decorator
def spectroscopy_multi(Sample, ctrl=[],freq=6.0*GHz, sb_freq=0*MHz,
                 specLen=1*us, specAmp=0.05, bias=None, zpa=0,
                 name='spectroscopy', des=''):
    """ 
    output xy pulse in all qubits
    and measure together, 
    search qubits f10 at same time
    """
    sample, qubits, Qubits = loadQubits(Sample, write_access=True)

    for qb in qubits:
        qb['qa_demod_freq'] = (qb['readout_freq']-LO_ctrl(Sample,qb,'frequency',TYPE='readout'))

    # if bias is None:
    #     bias = q['bias']

    # set some parameters name;
    axes = [(freq, 'freq'), (sb_freq, 'sb_freq'), 
            (specLen, 'specLen'), (specAmp, 'specAmp'),
            (bias, 'bias'), (zpa, 'zpa')]
    deps = MultiChannel_Deps(qubits,raw=True,prob=True)
    kw = {}

    # create dataset
    dataset = sweeps.prepDataset(
        sample, name+' '+des, axes, deps, measure=list(range(len(qubits))), kw=kw)

    def runSweeper(devices, para_list):
        freq, sb_freq, specLen, specAmp, bias, zpa = para_list
        clear_qubitsChannels(qubits)
        if bias is not None:
            for qb in qubits:
                qb['bias'] = bias
        sample['Settings']['anritsu_xy_1']['frequency'] = freq*GHz

        ## start this sequence
        start = 0
        for qname in ctrl:
            qb = sample[qname]
            add_spectroscopyPulse(qb,specAmp,start,specLen,freq=sb_freq)
            add_zpaPulse(qb,zpa,start,specLen)
        start += specLen

        ## set readout pulse & demod window
        for qb in qubits:
            qb['do_readout'] = True
            add_readoutPulse(qb,start,sample,SyncDemod=True)
        ## set bias for qubits
        add_qubitsDC(qubits,sample)

        data = runQubits(qubits,sample['Settings'])

        result = MultiChannel_Result(qubits,data,raw=True,prob=True)
        return result

    axes_scans = gridSweep(axes)
    result_list = RunAllExperiment(runSweeper, axes_scans, dataset)











