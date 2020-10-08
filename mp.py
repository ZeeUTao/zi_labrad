# -*- coding: utf-8 -*-
"""
Created on 2020.09.09 20:57
Controller for Zurich Instruments
@author: , Huang Wenhui
"""

from functools import wraps
import logging # python standard module for logging facility
import zhinst.utils ## create API object
import textwrap ## to write sequencer's code
import time ## show total time in experiments
import matplotlib.pyplot as plt ## give picture
import numpy as np 
from numpy import pi
from math import ceil
import itertools
import scipy.optimize
# from zurich_qa import zurich_qa ## quantum analyzer class
# from zurich_hd import zurich_hd ## hdawg instruments class
# from microwave_source import microwave_source ## control microwave source by visa
import pyvisa
from importlib import reload
from zurichHelper import check_device,stop_device,mpAwg_init
from conf import loadInfo
from conf import qa,hd,mw,mw_r
# qa,hd,mw,mw_r are instances


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
from pyle.sweeps import checkAbort
import adjuster

from labrad.units import Unit,Value
_unitSpace = ('V','mV','us','ns','s','GHz','MHz','kHz','Hz','dBm','rad','None')
V, mV, us, ns,s, GHz, MHz,kHz,Hz, dBm, rad,_l  = [Unit(s) for s in _unitSpace]
ar = st.r
plt.ion()

cxn=labrad.connect()
dv = cxn.data_vault
# # specify the sample, in registry   
# from BatchRun import ss
# ss = None


logging.basicConfig(format='%(asctime)s | %(name)s [%(levelname)s] : %(message)s',
                    level=logging.INFO
                    )
"""
logging setting
"""



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
    logging.info(dataset.dependents)
    logging.info(dataset.independents)
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


def mpfunc_decorator(func):
    """
    do some stuff before call the function (func) in our experiment
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__name__)
        
        check_device()
        _t0_ = time.time()

        start_ts = time.time()
        result = func(*args, **kwargs)
        
        stop_device() ## stop all device running
        logger.info('use time (s): %.2f '%(time.time()-_t0_))
        return result
    return wrapper


def runQ(qubits,hd,qa):
    # generally for running multiqubits
    ## reload new waveform in this runQ
    for q in qubits: ## 多比特还需要修改这个send方法; 
        hd.send_waveform(waveform=q.xy+q.dc+q.z,ports=[q.channels['xy_I'],q.channels['xy_Q'],q.channels['dc'],q.channels['z']])
        qa.send_waveform(waveform=q.r)
    ## start to run experiment
    hd.awg_open()
    qa.awg_open()
    data = qa.get_data()
    return data

@mpfunc_decorator
def s21_scan(sample,measure=0,stats=1024,freq=ar[6.4:6.9:0.002,GHz],delay=0*ns,
    mw_power=None,bias=None,power=None,sb_freq=None,
    name='s21_scan',des='',back=False,noisy=True):
    """ 
    s21 scanning

    Args:
        sample: select experimental parameter from registry;
        stats: Number of Samples for one sweep point;
    """

    ## load parameters 
    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    q = qubits[measure]
    q.ch = dict(q['channels'])
    q.channels = dict(q['channels'])

    if bias == None:
        bias = q['bias']
    if power == None:
        power = q['readout_amp']
    if sb_freq == None:
        q.demod_freq = q['readout_freq'][Hz]-q['readout_mw_fc'][Hz]
    else:
        q.demod_freq = sb_freq[Hz]
    if mw_power == None:
        mw_power = q['readout_mw_power']
    q.awgs_pulse_len += np.max(delay) ## add max length of hd waveforms 

    ## set some parameters name;
    axes = [(freq,'freq'),(bias,'bias'),(power,'power'),(sb_freq,'sb_freq'),(mw_power,'mw_power'),
            (delay,'delay')]
    deps = [('Amplitude','s21 for','a.u.'),('Phase','s21 for','rad'),
                ('I','',''),('Q','','')]

    kw = {'stats': stats}

    # create dataset
    dataset = sweeps.prepDataset(sample, name+des, axes, deps,kw=kw)
    dataset_create(dv,dataset)


    w_qa,w_hd = mpAwg_init(q,stats)

    def runSweeper(para_list):
        freq,bias,power,sb_freq,mw_power,delay = para_list
        q.power_r = power2amp(power) ## consider 0.5*Vpp for 50 Ohm impedance

        ## set microwave source device
        q['readout_mw_fc'] = (freq - q.demod_freq)*Hz
        mw_r.set_freq(q['readout_mw_fc'][Hz])
        mw_r.set_power(mw_power)

        ## write waveforms 
        start = 0   ## prepare xy/z pulse, control sequence by 'start'
        ## 如果有xy/z pulse,就在这加,然后更改start;
        q.z = [w_hd.square(amp=0)]
        q.xy = [w_hd.square(amp=0),w_hd.square(amp=0)]
        ## 确保hd和qa部分的pulse都可以被正确bias;
        q.dc = [w_hd.bias(amp=bias)]
        
        start += delay
        start += 100e-9 ## 额外添加的读取间隔,避免hd下降沿对读取部分的影响;
        start += q['qa_start_delay'][s] ## 修正hd与qa之间trigger导致的时序不准,正确设置QA pulse和demod窗口的启动时刻
        ## 记录hd部分涉及的pulse length; 
        if hd.pulse_length_s != start:
            hd.pulse_length_s = start
            qa.set_adc_trig_delay(q['bias_start'][s]+hd.pulse_length_s)

        ## 结束hd脉冲,开始设置读取部分
        q.demod_phase = q.qa_adjusted_phase[Hz]*(qa.adc_trig_delay_s) ## adjusted phase -> unit[MHz]
        q.r = w_qa.readout([q])

        ## start this runQ Experiment
        data = runQ([q],hd,qa)

        ## analyze data and return
        for _d_ in data:
            amp = np.mean(np.abs(_d_))/q.power_r ## unit: dB; only relative strength;
            phase = np.mean(np.angle(_d_))
            Iv = np.mean(np.real(_d_))
            Qv = np.mean(np.imag(_d_))
        ## multiply channel should unfold to a list for return result
        result = [amp,phase,Iv,Qv]
        return result 

    result_list = []
    ## start to running ##
    axes_scans = checkAbort(gridSweep(axes), prefix=[1],func=stop_device)
    for axes_scan in axes_scans:
        _para_ = [Unit2SI(a) for a in axes_scan[0]]
        indep = [Unit2num(a) for a in axes_scan[1]]

        result = runSweeper(_para_)
        
        data_send = indep + result
        if noisy:
            print(', '.join([format(x,'f') for x in data_send])) ## show sweeping data
        result_list.append(data_send)
        dv.add(data_send.copy()) ## save value to dataVault


    if back:
        return result_list,q,w_hd,w_qa



@mpfunc_decorator
def spectroscopy(sample,measure=0,stats=1024,freq=ar[6.0:4.5:0.005,GHz],specLen=1*us,specAmp=0.1,sb_freq=0*Hz,
    bias=None,zpa=None,
    name='spectroscopy',des='',back=False,noisy=True):
    """ 
        sample: select experimental parameter from registry;
        stats: Number of Samples for one sweep point;
    """
    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    q = qubits[measure]
    q.channels = dict(q['channels'])

    if bias == None:
        bias = q['bias']
    if zpa == None:
        zpa = q['zpa']
    q.power_r = power2amp(q['readout_amp']['dBm'])
    q.demod_freq = (q['readout_freq']-q['readout_mw_fc'])[Hz]

    ## set some parameters name;
    axes = [(freq,'freq'),(specAmp,'specAmp'),(bias,'bias'),(zpa,'zpa')]
    deps = [('Amplitude','s21 for','a.u.'),('Phase','s21 for','rad'),
            ('I','',''),('Q','',''),
            ('probability |0>','',''),
            ('probability |1>','','')]

    kw = {'stats': stats,'sb_freq': sb_freq}

    # create dataset
    dataset = sweeps.prepDataset(sample, name+des, axes, deps,kw=kw)
    dataset_create(dv,dataset)

    w_qa,w_hd = mpAwg_init(q,stats)

    def runSweeper(para_list):
        freq,specAmp,bias,zpa = para_list
        ## set device parameter
        mw.set_freq(freq-sb_freq[Hz])
        ## write waveforms 
        start = 0
        ## 如果有xy/z pulse,就在这加,然后更改start;
        q.z = [w_hd.square(amp=zpa,start=start,end=start+specLen[s]+100e-9)]
        start += 50e-9
        q.xy = [w_hd.cosine(amp=specAmp,freq=sb_freq[Hz],start=start,end=start+specLen[s]),
                w_hd.sine(amp=specAmp,freq=sb_freq[Hz],start=start,end=start+specLen[s])]

        start += specLen[s] + 50e-9
        ## 确保hd和qa部分的pulse都可以被正确bias;
        q.dc = [w_hd.bias(amp=bias)]

        start += 100e-9 ## 额外添加的读取间隔,避免hd下降沿对读取部分的影响;
        start += q['qa_start_delay'][s] ## 修正hd与qa之间trigger导致的时序不准,正确启动QA pulse和demod窗口
        ## 记录hd部分涉及的pulse length; 
        if hd.pulse_length_s != start:
            hd.pulse_length_s = start
            qa.set_adc_trig_delay(q['bias_start'][s]+hd.pulse_length_s)

        ## 结束hd脉冲,开始设置读取部分
        q.r = w_qa.readout([q])

        ## start to run experiment
        data = runQ([q],hd,qa)

        ## analyze data and return
        for _d_ in data:
            amp = np.mean(np.abs(_d_))/q.power_r ##20*np.log10(np.mean(np.abs(_d_))/power) ## unit: dB; only relative strength;
            phase = np.mean(np.angle(_d_))
            Iv = np.mean(np.real(_d_))
            Qv = np.mean(np.imag(_d_))
            prob = tunneling([q],[_d_],level=2)
        ## multiply channel should unfold to a list for return result
        result = [amp,phase,Iv,Qv,prob[0],prob[1]]
        return result

 
    result_list = []
    ## start to running ##
    axes_scans = checkAbort(gridSweep(axes), prefix=[1],func=stop_device)
    for axes_scan in axes_scans:
        _para_ = [Unit2SI(a) for a in axes_scan[0]]
        indep = [Unit2num(a) for a in axes_scan[1]]

        result = runSweeper(_para_)
        
        data_send = indep + result
        if noisy:
            print(', '.join([format(x,'.3f') for x in data_send])) ## show in scientific notation
        result_list.append(data_send)
        dv.add(data_send.copy()) ## save value to dataVault

    if back:
        return result_list,q



@mpfunc_decorator
def rabihigh(sample,measure=0,stats=1024,piamp=ar[0:1:0.02],df=0*MHz,
    bias=None,zpa=None,
    name='rabihigh',des='',back=False,noisy=True):
    """ 
        sample: select experimental parameter from registry;
        stats: Number of Samples for one sweep point;
    """
    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    q = qubits[measure]
    q.channels = dict(q['channels'])

    if bias == None:
        bias = q['bias']
    if zpa == None:
        zpa = q['zpa']
    if piamp == None:
        piamp = q['piAmp']
    q.power_r = power2amp(q['readout_amp']['dBm'])
    q.demod_freq = q['readout_freq'][Hz]-q['readout_mw_fc'][Hz]
    q.sb_freq = (q['f10'] - q['xy_mw_fc'])[Hz]

    ## set some parameters name;
    axes = [(bias,'bias'),(zpa,'zpa'),(df,'df'),(piamp,'piamp')]
    deps = [('Amplitude','s21 for','a.u.'),('Phase','s21 for','rad'),
            ('I','',''),('Q','',''),
            ('probability','|0>',''),
            ('probability','|1>','')]

    kw = {'stats': stats}

    # create dataset
    dataset = sweeps.prepDataset(sample, name+des, axes, deps,kw=kw)
    dataset_create(dv,dataset)

    w_qa,w_hd = mpAwg_init(q,stats)

    q_copy = q.copy()
    def runSweeper(para_list):
        bias,zpa,df,piamp = para_list
        q['xy_mw_fc'] = q_copy['xy_mw_fc'] + df*Hz

        ## set device parameter
        mw.set_freq(q['xy_mw_fc'][Hz])
        ## write waveforms 
        start = 0
        ## 如果有xy/z pulse,就在这加,然后更改start;
        q.z = [w_hd.square(amp=zpa,start=start,length=q.piLen[s]+100e-9)]
        start += 50e-9
        q.xy = [w_hd.cosine(amp=piamp,freq=q.sb_freq+df,start=start,length=q.piLen[s]),
                w_hd.sine(amp=piamp,freq=q.sb_freq+df,start=start,length=q.piLen[s])]

        start += q.piLen[s] + 50e-9
        ## 确保hd和qa部分的pulse都可以被正确bias;
        q.dc = [w_hd.bias(amp=bias)]

        start += 100e-9 ## 额外添加的读取间隔,避免hd下降沿对读取部分的影响;
        start += q['qa_start_delay'][s] ## 修正hd与qa之间trigger导致的时序不准,正确启动QA pulse和demod窗口
        ## 记录hd部分涉及的pulse length; 
        if hd.pulse_length_s != start:
            hd.pulse_length_s = start
            qa.set_adc_trig_delay(q['bias_start'][s]+hd.pulse_length_s)

        ## 结束hd脉冲,开始设置读取部分
        q.demod_phase = q.qa_adjusted_phase[Hz]*(qa.adc_trig_delay_s) ## adjusted phase -> unit[MHz]
        q.r = w_qa.readout([q])

        ## start to run experiment
        data = runQ([q],hd,qa)

        ## analyze data and return
        for _d_ in data:
            amp = np.mean(np.abs(_d_))/q.power_r ##20*np.log10(np.mean(np.abs(_d_))/power) ## unit: dB; only relative strength;
            phase = np.mean(np.angle(_d_))
            Iv = np.mean(np.real(_d_))
            Qv = np.mean(np.imag(_d_))
            prob = tunneling([q],[_d_],level=2)
        ## multiply channel should unfold to a list for return result
        result = [amp,phase,Iv,Qv,prob[0],prob[1]]
        return result

 
    result_list = []
    ## start to running ##
    axes_scans = checkAbort(gridSweep(axes), prefix=[1],func=stop_device)
    for axes_scan in axes_scans:
        _para_ = [Unit2SI(a) for a in axes_scan[0]]
        indep = [Unit2num(a) for a in axes_scan[1]]

        result = runSweeper(_para_)
        
        data_send = indep + result
        if noisy:
            print(', '.join([format(x,'.3f') for x in data_send])) ## show in scientific notation
        result_list.append(data_send)
        dv.add(data_send.copy()) ## save value to dataVault

    if back:
        return result_list,q




@mpfunc_decorator
def IQraw(sample,measure=0,stats=1024,update=True,analyze=False,
    name='IQ raw',des='',back=False,noisy=True):
    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    q = qubits[measure]
    Qb = Qubits[measure]
    q.channels = dict(q['channels'])

    q.power_r = power2amp(q['readout_amp']['dBm'])
    q.demod_freq = q['readout_freq'][Hz]-q['readout_mw_fc'][Hz]
    q.sb_freq = (q['f10'] - q['xy_mw_fc'])[Hz]

    ## set some parameters name;
    axes = []
    deps = [('Is','|0>',''),('Qs','|0>',''),
            ('Is','|1>',''),('Qs','|1>','')]
    kw = {'stats': stats}

    # create dataset
    dataset = sweeps.prepDataset(sample, name+des, axes, deps,kw=kw)
    dataset_create(dv,dataset)

    w_qa,w_hd = mpAwg_init(q,stats)

    def runSweeper():
        # ## set device parameter
        # mw.set_freq(q['xy_mw_fc'][Hz])

        ## with pi pulse --> |1> ##
        start = 0
        ## 如果有xy/z pulse,就在这加,然后更改start;
        q.z = [w_hd.square(amp=q.zpa[V],start=start,length=q.piLen[s]+100e-9)]
        start += 50e-9
        q.xy = [w_hd.cosine(amp=q.piAmp,freq=q.sb_freq,start=start,length=q.piLen[s]),
                w_hd.sine(amp=q.piAmp,freq=q.sb_freq,start=start,length=q.piLen[s])]
        start += q.piLen[s] + 50e-9
        ## 确保hd和qa部分的pulse都可以被正确bias;
        q.dc = [w_hd.bias(amp=q.bias[V])]

        start += 100e-9 ## 额外添加的读取间隔,避免hd下降沿对读取部分的影响;
        start += q['qa_start_delay'][s] ## 修正hd与qa之间trigger导致的时序不准,正确启动QA pulse和demod窗口
        ## 记录hd部分涉及的pulse length; 
        if hd.pulse_length_s != start:
            hd.pulse_length_s = start
            qa.set_adc_trig_delay(q['bias_start'][s]+hd.pulse_length_s)

        ## 结束hd脉冲,开始设置读取部分
        q.demod_phase = q.qa_adjusted_phase[Hz]*(qa.adc_trig_delay_s) ## adjusted phase -> unit[MHz]
        q.r = w_qa.readout([q])

        ## start to run experiment
        data1 = runQ([q],hd,qa)

        ## no pi pulse --> |0> ##
        q.xy = [w_hd.square(amp=0),w_hd.square(amp=0)]
        ## start to run experiment
        data0 = runQ([q],hd,qa)

        ## analyze data and return
        for d0 in data0:
            Is0 = np.real(d0)
            Qs0 = np.imag(d0)

        for d1 in data1:
            Is1 = np.real(d1)
            Qs1 = np.imag(d1)




        ## multiply channel should unfold to a list for return result
        result = [[Is0],[Qs0],[Is1],[Qs1]]
        return np.vstack(result).T

 
    result_list = []
    ## start to running ##
    axes_scans = checkAbort(gridSweep(axes), prefix=[1],func=stop_device)
    for axes_scan in axes_scans:
        # _para_ = [Unit2SI(a) for a in axes_scan[0]]
        # indep = [Unit2num(a) for a in axes_scan[1]]

        result = runSweeper()
        
        data_send = result
        # if noisy:
        #     print(', '.join([format(x,'.3f') for x in data_send])) ## show in scientific notation
        result_list = data_send
        dv.add(data_send.copy()) ## save value to dataVault

    Is0, Qs0, Is1, Qs1 = result_list.T
    if update:
        Qb['center|0>'] = [np.mean(Is0),np.mean(Qs0)]
        Qb['center|1>'] = [np.mean(Is1),np.mean(Qs1)]
        adjuster.IQ_center(qubit=Qb, data=result_list)

    center0 = Qb['center|0>'][0] + 1j*Qb['center|0>'][1]
    center1 = Qb['center|1>'][0] + 1j*Qb['center|1>'][1]

    if analyze:
        sig0s, sig1s = rotateData(result_list, center0, center1,doPlot=True)
        result = anaSignal(sig0s, sig1s, stats, labels=['|0>','|1>'], fignum=[102, 103])

        plt.figure(101)
        theta = np.linspace(0, 2*np.pi, 100)
        for idx in range(3):
            plt.plot(np.real(center0+(idx+1)*result[1]*np.exp(1j*theta)),np.imag(center0+(idx+1)*result[1]*np.exp(1j*theta)),'g-',lw=2)
            plt.plot(np.real(center1+(idx+1)*result[2]*np.exp(1j*theta)),np.imag(center1+(idx+1)*result[2]*np.exp(1j*theta)),'k-',lw=2)
        sepPoint = center0+np.cos(np.angle(center1-center0))*result[-1]+1j*np.sin(np.angle(center1-center0))*result[-1]
        sepPoint1 = sepPoint+np.cos(np.angle(center1-center0)+np.pi/2)*50.0+1j*np.sin(np.angle(center1-center0)+np.pi/2)*50.0
        sepPoint2 = sepPoint+np.cos(np.angle(center1-center0)-np.pi/2)*50.0+1j*np.sin(np.angle(center1-center0)-np.pi/2)*50.0
        plt.plot(np.real(np.array([sepPoint1,sepPoint,sepPoint2])), np.imag(np.array([sepPoint1,sepPoint,sepPoint2])), 'g-', lw=2)

    if back:
        return result,q


@mpfunc_decorator
def T1_visibility(sample,measure=0,stats=1024,delay=ar[0:12:0.5,us],
    zpa=None,bias=None,
    name='T1_visibility',des='',back=False,noisy=True):
    """ sample: select experimental parameter from registry;
        stats: Number of Samples for one sweep point;
    """
    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    q = qubits[measure]
    q.channels = dict(q['channels'])

    if bias == None:
        bias = q['bias']
    if zpa == None:
        zpa = q['zpa']
    q.power_r = power2amp(q['readout_amp']['dBm'])
    q.demod_freq = q['readout_freq'][Hz]-q['readout_mw_fc'][Hz]
    q.sb_freq = (q['f10'] - q['xy_mw_fc'])[Hz]
    q.awgs_pulse_len += np.max(delay) ## add max length of hd waveforms 

    ## set some parameters name;
    axes = [(bias,'bias'),(zpa,'zpa'),(delay,'delay')]
    deps = [('Amplitude','1','a.u.'),
            ('Phase','1','rad'),
            ('prob with pi pulse','|1>',''),
            ('Amplitude','0','a.u.'),
            ('Phase','0','rad'),
            ('prob without pi pulse','|1>','')]

    kw = {'stats': stats}

    # create dataset
    dataset = sweeps.prepDataset(sample, name+des, axes, deps,kw=kw)
    dataset_create(dv,dataset)

    w_qa,w_hd = mpAwg_init(q,stats)

    def runSweeper(para_list):
        bias,zpa,delay = para_list
        # ## set device parameter
        # mw.set_freq(q['xy_mw_fc'][Hz])

        ### ----- with pi pulse ----- ###
        start = 0  
        ## 如果有xy/z pulse,就在这加,然后更改start;
        q.z = [w_hd.square(amp=zpa,start=start,length=delay+q.piLen[s]+100e-9)]
        start += 50e-9
        q.xy = [w_hd.cosine(amp=q.piAmp,freq=q.sb_freq,start=start,length=q.piLen[s]),
                w_hd.sine(amp=q.piAmp,freq=q.sb_freq,start=start,length=q.piLen[s])]
        start += q.piLen[s] + delay + 50e-9
        ## 确保hd和qa部分的pulse都可以被正确bias;
        q.dc = [w_hd.bias(amp=bias)]

        start += 100e-9 ## 额外添加的读取间隔,避免hd下降沿对读取部分的影响;
        start += q['qa_start_delay'][s] ## 修正hd与qa之间trigger导致的时序不准,正确启动QA pulse和demod窗口
        ## 记录hd部分涉及的pulse length; 
        if hd.pulse_length_s != start:
            hd.pulse_length_s = start
            qa.set_adc_trig_delay(q['bias_start'][s]+hd.pulse_length_s)

        ## 结束hd脉冲,开始设置读取部分
        q.demod_phase = q.qa_adjusted_phase[Hz]*(qa.adc_trig_delay_s) ## adjusted phase -> unit[MHz]
        q.r = w_qa.readout([q])

        ## start to run experiment
        data1 = runQ([q],hd,qa)
        ## analyze data and return
        for _d_ in data1:
            amp1 = np.mean(np.abs(_d_))/q.power_r ## unit: dB; only relative strength;
            phase1 = np.mean(np.angle(_d_))
            prob1 = tunneling([q],[_d_],level=2)

        ### ----- without pi pulse ----- ###
        q.xy = [w_hd.square(amp=0),w_hd.square(amp=0)]
        ## start to run experiment
        data0 = runQ([q],hd,qa)
        ## analyze data and return
        for _d_ in data0:
            amp0 = np.mean(np.abs(_d_))/q.power_r ## unit: dB; only relative strength;
            phase0 = np.mean(np.angle(_d_))
            prob0 = tunneling([q],[_d_],level=2)


        ## multiply channel should unfold to a list for return result
        result = [amp1,phase1,prob1[1],amp0,phase0,prob0[1]]
        return result

 
    result_list = []
    ## start to running ##
    axes_scans = checkAbort(gridSweep(axes), prefix=[1],func=stop_device)
    for axes_scan in axes_scans:
        _para_ = [Unit2SI(a) for a in axes_scan[0]]
        indep = [Unit2num(a) for a in axes_scan[1]]

        result = runSweeper(_para_)
        
        data_send = indep + result
        if noisy:
            print(', '.join([format(x,'.3f') for x in data_send])) ## show in scientific notation
        result_list.append(data_send)
        dv.add(data_send.copy()) ## save value to dataVault


    if back:
        return result_list,q





@mpfunc_decorator
def ramsey(sample,measure=0,stats=1024,delay=ar[0:10:0.4,us],
    repetition=1,df=0*MHz,fringeFreq=10*MHz,PHASE=0,bias=0.0,
    name='ramsey',des='',back=False,noisy=True):
    """ sample: select experimental parameter from registry;
        stats: Number of Samples for one sweep point;
    """
    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    q = qubits[measure]
    q.channels = dict(q['channels'])


    q.power_r = power2amp(q['readout_amp']['dBm'])
    q.demod_freq = q['readout_freq'][Hz]-q['readout_mw_fc'][Hz]
    q.sb_freq = (q['f10'] - q['xy_mw_fc'])[Hz]
    q.awgs_pulse_len += np.max(delay) ## add max length of hd waveforms 

    ## set some parameters name;
    axes = [(repetition, 'repetition'),(delay,'delay'),(df,'df'),(fringeFreq,'fringeFreq'),(PHASE,'PHASE')]
    deps = [('Amplitude','s21 for','a.u.'),('Phase','s21 for','rad'),
            ('I','',''),('Q','',''),('prob |0>','',''),('prob |1>','','')]

    kw = {'stats': stats,
          'fringeFreq': fringeFreq}

    # create dataset
    dataset = sweeps.prepDataset(sample, name+des, axes, deps,kw=kw)
    dataset_create(dv,dataset)

    w_qa,w_hd = mpAwg_init(q,stats)

    q_copy = q.copy()
    def runSweeper(para_list):
        repetition,delay,df,fringeFreq,PHASE = para_list
        ## set device parameter
        q['xy_mw_fc'] = q_copy['xy_mw_fc'] + df*Hz
        mw.set_freq(q['xy_mw_fc'][Hz])
        bias = q.bias['V']

        ### ----- begin waveform ----- ###
        start = 0  
        ## 如果有xy/z pulse,就在这加,然后更改start;
        q.z = [w_hd.square(amp=q.zpa[V],start=start,length=delay+2*q.piLen[s]+100e-9)]
        start += 50e-9
        q.xy = [w_hd.cosine(amp=q.piAmp/2,freq=q.sb_freq,phase=0,start=start,length=q.piLen[s]),
                w_hd.sine(amp=q.piAmp/2,freq=q.sb_freq,phase=0,start=start,length=q.piLen[s])]
        start += delay + q.piLen[s]
        q.xy = [w_hd.cosine(amp=q.piAmp/2,freq=q.sb_freq,phase=2*np.pi*fringeFreq*delay+PHASE,start=start,length=q.piLen[s]),
                w_hd.sine(amp=q.piAmp/2,freq=q.sb_freq,phase=2*np.pi*fringeFreq*delay+PHASE,start=start,length=q.piLen[s])]
        start += q.piLen[s] + 50e-9
        ## 确保hd和qa部分的pulse都可以被正确bias;
        q.dc = [w_hd.bias(amp=bias)]

        start += 100e-9 ## 额外添加的读取间隔,避免hd下降沿对读取部分的影响;
        start += q['qa_start_delay'][s] ## 修正hd与qa之间trigger导致的时序不准,正确启动QA pulse和demod窗口
        ## 记录hd部分涉及的pulse length; 
        if hd.pulse_length_s != start:
            hd.pulse_length_s = start
            qa.set_adc_trig_delay(q['bias_start'][s]+hd.pulse_length_s)

        ## 结束hd脉冲,开始设置读取部分
        q.demod_phase = q.qa_adjusted_phase[Hz]*(qa.adc_trig_delay_s) ## adjusted phase -> unit[MHz]
        q.r = w_qa.readout([q])

        ## start to run experiment
        data = runQ([q],hd,qa)
        ## analyze data and return
        for _d_ in data:
            amp = np.mean(np.abs(_d_))/q.power_r ## unit: dB; only relative strength;
            phase = np.mean(np.angle(_d_))
            Iv = np.mean(np.real(_d_))
            Qv = np.mean(np.imag(_d_))
            prob = tunneling([q],[_d_],level=2)

        ## multiply channel should unfold to a list for return result
        result = [amp,phase,Iv,Qv,prob[0],prob[1]]
        return result

 
    result_list = []
    ## start to running ##
    axes_scans = checkAbort(gridSweep(axes), prefix=[1],func=stop_device)
    for axes_scan in axes_scans:
        _para_ = [Unit2SI(a) for a in axes_scan[0]]
        indep = [Unit2num(a) for a in axes_scan[1]]

        result = runSweeper(_para_)
        
        data_send = indep + result
        if noisy:
            print(', '.join([format(x,'.3f') for x in data_send])) ## show in scientific notation
        result_list.append(data_send)
        dv.add(data_send.copy()) ## save value to dataVault
    if back:
        return result_list,q



@mpfunc_decorator
def s21_dispersiveShift(sample,measure=0,stats=1024,freq=ar[6.4:6.9:0.002,GHz],delay=0*ns,
    mw_power=None,bias=None,power=None,sb_freq=None,
    name='s21_disperShift',des='',back=False,noisy=True):
    """ 
        sample: select experimental parameter from registry;
        stats: Number of Samples for one sweep point;
    """
    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    q = qubits[measure]
    q.channels = dict(q['channels'])

    if bias == None:
        bias = q['bias']
    if power == None:
        power = q['readout_amp']
    if sb_freq == None:
        sb_freq = q['readout_freq'][Hz]-q['readout_mw_fc'][Hz]
    if mw_power == None:
        mw_power = q['readout_mw_power']
    q.awgs_pulse_len += np.max(delay) ## add max length of hd waveforms 

    ## set some parameters name;
    axes = [(freq,'freq'),(bias,'bias'),(power,'power'),(sb_freq,'sb_freq'),(mw_power,'mw_power'),
            (delay,'delay')]
    deps = [('Amplitude|0>','S11 for %s'%q.__name__,''),('Phase|0>', 'S11 for %s'%q.__name__, rad)]
    deps.append(('I|0>','',''))
    deps.append(('Q|0>','',''))
    
    deps.append(('Amplitude|1>', 'S11 for %s'%q.__name__, rad))
    deps.append(('Phase|1>', 'S11 for %s'%q.__name__, rad))
    deps.append(('I|1>','',''))
    deps.append(('Q|1>','',''))
    deps.append(('SNR','',''))

    kw = {'stats': stats}

    # create dataset
    dataset = sweeps.prepDataset(sample, name+des, axes, deps,kw=kw)
    dataset_create(dv,dataset)

    w_qa,w_hd = mpAwg_init(q,stats)

    def runSweeper(para_list):
        freq,bias,power,sb_freq,mw_power,delay = para_list
        q.power_r = power2amp(power) ## consider 0.5*Vpp for 50 Ohm impedance

        ## set microwave source device
        q['readout_mw_fc'] = (freq - q.demod_freq)*Hz
        mw_r.set_freq(q['readout_mw_fc'][Hz])
        mw_r.set_power(mw_power)

        ## write waveforms 
        ## with pi pulse --> |1> ##
        q.xy_sb_freq = (q['f10'] - q['xy_mw_fc'])[Hz]
        start = 0 
        q.z = [w_hd.square(amp=q.zpa[V],start=start,length=q.piLen[s]+100e-9)]
        start += 50e-9
        q.xy = [w_hd.cosine(amp=q.piAmp,freq=q.xy_sb_freq,start=start,length=q.piLen[s]),
                w_hd.sine(amp=q.piAmp,freq=q.xy_sb_freq,start=start,length=q.piLen[s])]
        start += q.piLen[s] + 50e-9
        q.dc = [w_hd.bias(amp=bias)]
        
        start += delay
        start += 100e-9 ## 额外添加的读取间隔,避免hd下降沿对读取部分的影响;
        start += q['qa_start_delay'][s] ## 修正hd与qa之间trigger导致的时序不准,正确设置QA pulse和demod窗口的启动时刻
        ## 记录hd部分涉及的pulse length; 
        if hd.pulse_length_s != start:
            hd.pulse_length_s = start
            qa.set_adc_trig_delay(q['bias_start'][s]+hd.pulse_length_s)

        ## 结束hd脉冲,开始设置读取部分
        q.demod_phase = q.qa_adjusted_phase[Hz]*(qa.adc_trig_delay_s) ## adjusted phase -> unit[MHz]
        q.r = w_qa.readout([q])

        ## start to run experiment
        data1 = runQ([q],hd,qa)

        ## no pi pulse --> |0> ##
        q.xy = [w_hd.square(amp=0),w_hd.square(amp=0)]
        ## start to run experiment
        data0 = runQ([q],hd,qa)

        ## analyze data and return
        for _d_ in data0:
            amp0 = np.mean(np.abs(_d_))/q.power_r ## unit: dB; only relative strength;
            phase0 = np.mean(np.angle(_d_))
            Iv0 = np.mean(np.real(_d_))
            Qv0 = np.mean(np.imag(_d_))
        for _d_ in data1:
            amp1 = np.mean(np.abs(_d_))/q.power_r ## unit: dB; only relative strength;
            phase1 = np.mean(np.angle(_d_))
            Iv1 = np.mean(np.real(_d_))
            Qv1 = np.mean(np.imag(_d_))
        ## multiply channel should unfold to a list for return result
        result = [amp0,phase0,Iv0,Qv0]
        result += [amp1,phase1,Iv1,Qv1]
        result += [np.abs((Iv1-Iv0)+1j*(Qv1-Qv0))]
        return result 

    result_list = []
    ## start to running ##
    axes_scans = checkAbort(gridSweep(axes), prefix=[1],func=stop_device)
    for axes_scan in axes_scans:
        _para_ = [Unit2SI(a) for a in axes_scan[0]]
        indep = [Unit2num(a) for a in axes_scan[1]]

        result = runSweeper(_para_)
        
        data_send = indep + result
        if noisy:
            print(', '.join([format(x,'f') for x in data_send])) ## show in scientific notation
        result_list.append(data_send)
        dv.add(data_send.copy()) ## save value to dataVault
    if back:
        return result_list,q,w_hd,w_qa


















#### ----- dataprocess tools ----- ####

def tunneling(qubits,data,level=2):
    ## generated to N 20190618 -- ZiyuTao
    # q1 q2 q3  .... 
    qNum = len(qubits)
    counts_num = len(data[0])
    binary_count = np.zeros((counts_num),dtype=float)

    def get_meas(data0,q,Nq=level):
        # if measure 1 then return 1 
        sigs = data0
        
        total = len(sigs)
        distance = np.zeros((total,Nq))
        for i in np.arange(Nq):
            center_i = q['center|'+str(i)+'>'][0] + 1j*q['center|'+str(i)+'>'][1]
            distance_i = np.abs(sigs - center_i)
            distance[:,i]=  distance_i
        tunnels = np.zeros((total,))
        for i in np.arange(total):
            distancei = distance[i]
            tunneli = np.int(np.where(distancei == np.min(distancei))[0])
            tunnels[i] = tunneli 
        return tunnels

    for i in np.arange(qNum):
        binary_count += get_meas(data[i],qubits[i]) * (level**(qNum-1-i)) 
        
    res_store = np.zeros((level**qNum))
    for i in np.arange(level**qNum):
        res_store[i] = np.sum(binary_count == i) 
        
    prob = res_store/counts_num
    return prob


def rotateData(data, center0, center1, labels=['|0>','|1>'], doPlot=True):
    center = (center0+center1)/2.0
    theta = np.angle(center0-center1)
    I0s, Q0s, I1s, Q1s = data.T
    sig0s = I0s + 1j*Q0s
    sig1s = I1s + 1j*Q1s
    if doPlot:
        plt.figure(101)
        plt.plot(I0s, Q0s, 'b.', label=labels[0])
        plt.plot(I1s, Q1s, 'r.', label=labels[1])
        plt.plot((np.real(center0), np.real(center1)),(np.imag(center0), np.imag(center1)),'ko--',markersize=10,lw=3)
        plt.xlabel('I',size=20)
        plt.ylabel('Q',size=20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.legend()
        plt.subplots_adjust(bottom=0.125, left=0.15)
    sig0s = (sig0s-center)*np.exp(-1j*theta)
    sig1s = (sig1s-center)*np.exp(-1j*theta)
    return sig0s, sig1s

def fit_double_gaussian(count,xs,plot=True,output=True,color='black',method='bfgs'):
    def fitfunc(p,xs):
        return np.abs(p[0])*np.exp(-(xs-p[1])**2/(2*p[2]**2)) + np.abs(p[3])*np.exp(-(xs-p[4])**2/(2*p[5]**2))

    def errfunc(p):
        x = count - fitfunc(p,xs)
        return np.sum(x**2)
    
    xs_0 = xs[xs[:]>0]
    xs_1 = xs[xs[:]<0]
    count_0 = count[xs_1.shape[0]:]
    count_1 = count[0:xs_1.shape[0]]
    # out = minimize(errfunc,(np.max(count_0),np.mean(xs_0),(np.max(xs_0)-np.mean(xs_0))/2.0,np.max(count_1),np.mean(xs_1),(np.max(xs_1)-np.mean(xs_1))/2.0),
    # method = 'TNC', bounds = ((0.,6000.),(min(xs_1)/2.,max(xs_0)),(2.,max(xs_0)),(0.,6000.),(min(xs_1),max(xs_0)/2.),(2.,max(xs_0))))
    
    x0 = (np.max(count_0),np.mean(xs_0),(np.max(xs_0)-np.mean(xs_0))/2.0,np.max(count_1),np.mean(xs_1),(np.max(xs_1)-np.mean(xs_1))/2.0)
    bnds = ((0.0,6000.0),(0.0,max(xs_0)),(4.0,max(xs_0)),(0.0,6000.0),(min(xs_1),0.0),(4.0,max(xs_0)))
    # print x0
    # print bnds
    if method == 'bfgs':
        res = scipy.optimize.fmin_l_bfgs_b(errfunc,x0,bounds=bnds,approx_grad=True)[0]
    elif method == 'tnc':
        res = scipy.optimize.fmin_tnc(errfunc,x0,bounds=bnds,approx_grad=True)[0]
    else:
        res = scipy.optimize.fmin_slsqp(errfunc,x0,bounds=bnds)
        
    p = res
    if output:
        print('Amp|0>: ',np.abs(p[0]),'mean|0>: ',p[1],'deviation|0>: ',np.abs(p[2]))
        print('Amp|1>: ',np.abs(p[3]),'mean|1>: ',p[4],'deviation|1>: ',np.abs(p[5]))
    if plot:
        plt.plot(xs,fitfunc(p,xs),color=color,linewidth=2)
        
    return p

def anaSignal(sig0s, sig1s, stats, labels=['|0>','|1>'], fignum=[102, 103], debugFlag=False):
    stats0 = np.size(sig0s)
    stats1 = np.size(sig1s)

    if stats1 != stats:
        print('warning!!!!!!!!!!!!!!!!')
        print(stats0, stats1)

    plt.figure(fignum[0])
    plt.subplot(311)
    plt.plot(np.real(sig0s), np.imag(sig0s), 'bo', label=labels[0])
    plt.plot(np.real(sig1s), np.imag(sig1s), 'ro', label=labels[1],alpha=0.5)
    plt.legend()
    
    plt.subplot(312)
    sig_min = np.min([np.min(np.real(sig0s)), np.min(np.real(sig1s))])
    sig_max = np.max([np.max(np.real(sig0s)), np.max(np.real(sig1s))])
    counts0,positions0,patch0 = plt.hist(np.real(sig0s),range=(sig_min, sig_max),bins=50,alpha=0.5,label=labels[0])
    counts1,positions1,patch1 = plt.hist(np.real(sig1s),range=(sig_min, sig_max),bins=50,alpha=0.5,label=labels[1])
    positions0 = (positions0[0:-1] + positions0[1:])/2.0
    positions1 = (positions1[0:-1] + positions1[1:])/2.0
    
    result0 = fit_double_gaussian(counts0,positions0,color='blue')
    result1 = fit_double_gaussian(counts1,positions1,color='red')
    separation = result0[1]-result1[4]
    deviation0 = result0[2]
    deviation1 = result1[5]
    p00 = result0[:3]
    p01 = result0[3:]
    p11 = result1[3:]
    p10 = result1[:3]
    print('Separation: %.2f, SNR: %.2f'%(p00[1]-p11[1],(p00[1]-p11[1])/(p00[2]+p11[2])))
    def gaussianFunc(p,xs):
        return np.abs(p[0])*np.exp(-(xs-p[1])**2/(2*p[2]**2))
    def fitFunc(p,xs):
        return np.abs(p[0])*np.exp(-(xs-p[1])**2/(2*p[2]**2)) + np.abs(p[3])*np.exp(-(xs-p[4])**2/(2*p[5]**2))
    
    c0 = np.array([np.sum(counts0[:idx0]) for idx0 in range(len(positions0))])
    c1 = np.array([np.sum(counts1[:idx0]) for idx0 in range(len(positions0))])
    vis = np.array([-np.sum(counts0[:idx0])+np.sum(counts1[:idx0]) for idx0 in range(len(positions0))])
    idx = np.argmax(vis)
    vism = max(vis/float(stats))
    print('sepPoint: %.2f, Visibility: %.2f'%(positions0[idx], vism))
    
    plt.subplot(313)
    plt.plot(positions0, gaussianFunc(p00, positions0), 'b-', label=labels[0])
    plt.plot(positions0, gaussianFunc(p11, positions0), 'r-', label=labels[1])
    plt.fill_between(positions0, gaussianFunc(p00,positions0), where=(positions0<=positions0[idx-1]), color='b', alpha=0.5)
    plt.fill_between(positions0, gaussianFunc(p11,positions0), where=(positions0>=positions0[idx-1]), color='r', alpha=0.5)
    
    plt.figure(fignum[1])
    plt.plot(positions0, c0/float(stats), 'b-', lw=2,label='|0>')
    plt.plot(positions0, c1/float(stats), 'r-', lw=2,label='|1>')
    plt.plot(positions0, vis/float(stats), 'k-', lw=2)
    # plt.plot(positions0, vis/float(stats), 'r-', lw=1)
    plt.xlabel('Integration Axis', size=20)
    plt.ylabel('Visibility', size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.title('Maximum Visibility: %.2f'%np.max(vis/float(stats)))
    plt.subplots_adjust(bottom=0.125, left=0.18)

    separationError0 = np.sum(gaussianFunc(p00,positions0[:idx]))/np.sum(fitFunc(result0,positions0)) #calculate separation error
    separationError1 = np.sum(gaussianFunc(p11,positions0[idx:]))/np.sum(fitFunc(result1,positions0))
    Error0 = np.sum(counts0[1:idx])/np.float(stats)
    Error1 = np.sum(counts1[idx:])/np.float(stats)
    stateErrorp0 = Error0 - separationError0
    stateErrorp1 = Error1 - separationError1
    print('stateError|0>: %.1f%%, stateError|1>: %.1f%%'%(100*stateErrorp0, 100*stateErrorp1))
    print('Error|0>: %.1f%%, Error|1>: %.1f%%'%(100*Error0, 100*Error1))
    anaData = [separation, deviation0, deviation1, separationError0, separationError1, stateErrorp0, stateErrorp1, vism, positions0[idx], np.abs(positions0[idx]-result0[1])]
    
    if debugFlag:
        print(anaData)
        pdb.set_trace()
    return anaData 

#----------- dataprocess tools END -----------#




#------------- old code basket --------------#

#------------- code basket end --------------#
