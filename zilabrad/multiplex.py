# -*- coding: utf-8 -*-
"""
Controller for Zurich Instruments

The daily experiments are implemented via various function (s21_scan, ramsey...)

Created on 2020.09.09 20:57
@author: Huang Wenhui, Tao Ziyu
"""


import logging  # python standard module for logging facility
import time  # show total time in experiments
import matplotlib.pyplot as plt  # give picture
from functools import wraps
import functools
import numpy as np
from numpy import pi
import itertools

from zilabrad.instrument.qubitServer import stop_device
from zilabrad.instrument.qubitServer import RunAllExperiment as RunAllExp
from zilabrad.instrument.QubitContext import loadQubits, qubitContext
from zilabrad.instrument.qubitServer import runQubits as runQ


import zilabrad.plots.adjuster
import zilabrad.instrument.waveforms as waveforms
from zilabrad.pyle.envelopes import Envelope, NOTHING
from zilabrad.plots.dataProcess import datahelp
from zilabrad.plots import dataProcess

from zilabrad.pyle import sweeps
from zilabrad.pyle.util import sweeptools
from zilabrad.pyle.sweeps import checkAbort


from labrad.units import Unit, Value
import labrad
_unitSpace = ('V', 'mV', 'us', 'ns', 's', 'GHz',
              'MHz', 'kHz', 'Hz', 'dBm', 'rad', 'None')
V, mV, us, ns, s, GHz, MHz, kHz, Hz, dBm, rad, _l = [
    Unit(s) for s in _unitSpace]
ar = sweeptools.RangeCreator()

datahelper = datahelp()


def dataset_create(dataset):
    """Create the dataset."""
    dv = labrad.connect().data_vault
    dv.cd(dataset.path, dataset.mkdir)
    print(dataset.dependents)
    dv.new(dataset.name, dataset.independents, dataset.dependents)
    if len(dataset.params):
        dv.add_parameters(tuple(dataset.params))


_bringup_experiment = [
    's21_scan',
    'spectroscopy',
    'rabihigh',
    's21_dispersiveShift',
    'IQraw',
    'T1_visibility',
    'ramsey',
]


def _standard_exps(ss, funcs=_bringup_experiment):
    for func in funcs:
        expr = str(func) + '(ss)'
        eval(expr)
    return


def gridSweep(axes):
    """
    gridSweep generator yield all_paras, swept_paras
    if axes has one iterator, we can do a one-dimensional scanning
    if axes has two iterator, we can do a square grid scanning

    you can also create other generator, that is conditional for the result, do something like machnine-learning optimizer

    Example:
    axes = [(para1,'name1'),(para2,'name2'),(para3,'name3')...]
    para can be iterable or not-iterable

    for paras in gridSweep(axes):
        all_paras, swept_paras = paras

    all_paras: all parameters
    swept_paras: iterable parameters
    """
    if not len(axes):
        yield (), ()
    else:
        (param, _label), rest = axes[0], axes[1:]
        if np.iterable(param):  # TODO: different way to detect if something should be swept
            for val in param:
                for all, swept in gridSweep(rest):
                    yield (val,) + all, (val,) + swept
        else:
            for all, swept in gridSweep(rest):
                yield (param,) + all, swept


def expfunc_decorator(func):
    """
    do some stuff before call the function (func) in our experiment
    do stuffs.... func(*args) ... do stuffs
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_ts = time.time()
        try:
            result = func(*args, **kwargs)
        except KeyboardInterrupt:
            # stop in the middle
            print('KeyboardInterrupt')
            print('stop_device')
            timeNow = time.strftime("%Y-%m-%d %X", time.localtime())
            print(timeNow)
            stop_device()  # stop all device running

            return
        else:
            # finish in the end
            stop_device()
            timeNow = time.strftime("%Y-%m-%d %X", time.localtime())
            print(timeNow)
            return result
    return wrapper


def power2amp(power):
    """ 
    convert 'dBm' to 'V' for 50 Ohm (only value)
    Based on --> 0.5*Vpp=sqrt(2)*sqrt(50e-3*10^(dBm/10)); 
    """
    return 10**(power/20-0.5)


def amp2power(amp):
    """ 
    convert 'dBm' to 'V' for 50 Ohm (only value)
    Based on --> 0.5*Vpp=sqrt(2)*sqrt(50e-3*10^(dBm/10)); 
    """
    return 20*np.log10(amp)+10


def set_qubitsDC(qubits, experiment_length):
    for _qb in qubits:
        _qb['experiment_length'] = experiment_length
        _qb.dc = DCbiasPulse(_qb)
    return


def readoutPulse(q):
    amp = power2amp(q['readout_amp']['dBm'])
    length = q['readout_len']
    # when measuring T1_visibility or others, we need start delay of readout
    return waveforms.readout(amp=amp, phase=q['demod_phase'], freq=q['demod_freq'], start=0, length=length)


def DCbiasPulse(q):
    return waveforms.square(amp=q['bias'], start=-q['bias_start']['s'], end=q['bias_end']['s']+q['readout_len']['s']+q['experiment_length'])


def XYnothing(q):
    return [NOTHING, NOTHING]


def addXYgate(q, start, theta, phi):
    q.sb_freq = (q['f10'] - q['xy_mw_fc'])[Hz]
    amp = q.piAmp*theta/np.pi
    if 'xy' not in q:
        q['xy'] = XYnothing(q)

    q['xy'][0] += waveforms.cosine(amp=amp, freq=q.sb_freq,
                                   start=start, length=q.piLen[s], phase=phi)
    q['xy'][1] += waveforms.sine(amp=amp, freq=q.sb_freq,
                                 start=start, length=q.piLen[s], phase=phi)
    return


@expfunc_decorator
def s21_scan(sample, measure=0, stats=1024, freq=6.0*GHz, delay=0*ns, phase=0,
             mw_power=None, bias=None, power=None, zpa=0.0,
             name='s21_scan', des='', back=False):
    """ 
    s21 scanning

    Args:
        sample: select experimental parameter from registry;
        stats: Number of Samples for one sweep point;
    """
    # load parameters
    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    q = qubits[measure]
    q.channels = dict(q['channels'])
    q.stats = stats
    if freq == None:
        freq = q['readout_amp']
    if bias == None:
        bias = q['bias']
    if power == None:
        power = q['readout_amp']

    q.power_r = power2amp(q['readout_amp']['dBm'])
    q.demod_freq = q['readout_freq'][Hz]-q['readout_mw_fc'][Hz]
    if mw_power == None:
        mw_power = q['readout_mw_power']
    q.awgs_pulse_len += np.max(delay)  # add max length of hd waveforms

    # set some parameters name;
    axes = [(freq, 'freq'), (bias, 'bias'), (zpa, 'zpa'), (power, 'power'), (mw_power, 'mw_power'),
            (delay, 'delay'), (phase, 'phase')]
    deps = [('Amplitude', 's21 for', 'a.u.'), ('Phase', 's21 for', 'rad'),
            ('I', '', ''), ('Q', '', '')]
    kw = {'stats': stats}

    # create dataset
    dataset = sweeps.prepDataset(
        sample, name+des, axes, deps, measure=measure, kw=kw)

    def runSweeper(devices, para_list):
        freq, bias, zpa, power, mw_power, delay, phase = para_list
        q['readout_amp'] = power*dBm
        q.power_r = power2amp(power)

        q['readout_mw_fc'] = (freq - q['demod_freq'])*Hz
        start = 0
        q.z = waveforms.square(amp=zpa)
        q.xy = [waveforms.square(amp=0), waveforms.square(amp=0)]
        q['bias'] = bias

        start += delay
        start += 100e-9
        start += q['qa_start_delay']['s']

        for _qb in qubits:
            _qb['experiment_length'] = start
            _qb.dc = DCbiasPulse(_qb)

        q['do_readout'] = True
        q.r = readoutPulse(q)
        # q.demod_phase = q.qa_adjusted_phase[Hz]*(qa.adc_trig_delay_s) ## adjusted phase

        data = runQ([q], devices)

        _d_ = data[0]
        amp = np.abs(np.mean(_d_))/q.power_r
        phase = np.angle(np.mean(_d_))
        Iv = np.real(np.mean(_d_))
        Qv = np.imag(np.mean(_d_))
        return [amp, phase, Iv, Qv]

    axes_scans = checkAbort(gridSweep(axes), prefix=[1], func=stop_device)
    result_list = RunAllExp(runSweeper, axes_scans, dataset)
    if back:
        return result_list


@expfunc_decorator
def spectroscopy(sample, measure=0, stats=1024, freq=None, specLen=1*us, specAmp=0.05, sb_freq=None,
                 bias=None, zpa=None,
                 name='spectroscopy', des='', back=False):
    """ 
        sample: select experimental parameter from registry;
        stats: Number of Samples for one sweep point;
    """
    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    q = qubits[measure]
    q.channels = dict(q['channels'])
    q.stats = stats

    if freq == None:
        freq = q['f10']
    if bias == None:
        bias = q['bias']
    if zpa == None:
        zpa = q['zpa']
    if sb_freq == None:
        sb_freq = (q['f10'] - q['xy_mw_fc'])

    q.power_r = power2amp(q['readout_amp']['dBm'])
    q.demod_freq = (q['readout_freq']-q['readout_mw_fc'])[Hz]

    # set some parameters name;
    axes = [(freq, 'freq'), (specAmp, 'specAmp'),
            (specLen, 'specLen'), (bias, 'bias'), (zpa, 'zpa')]
    deps = dependents_1q()
    kw = {'stats': stats, 'sb_freq': sb_freq}

    for qb in qubits:
        # add max length of hd waveforms
        qb['awgs_pulse_len'] += np.max(specLen)
    # create dataset
    dataset = sweeps.prepDataset(
        sample, name+des, axes, deps, kw=kw, measure=measure)

    def runSweeper(devices, para_list):
        freq, specAmp, specLen, bias, zpa = para_list

        q['xy_mw_fc'] = freq*Hz-sb_freq

        start = 0
        q.z = waveforms.square(amp=zpa, start=start, length=specLen+100e-9)
        start += 50e-9
        q.xy = [waveforms.cosine(amp=specAmp, freq=sb_freq['Hz'], start=start, length=specLen),
                waveforms.sine(amp=specAmp, freq=sb_freq['Hz'], start=start, length=specLen)]
        start += specLen + 50e-9
        q['bias'] = bias

        start += 100e-9
        start += q['qa_start_delay']['s']

        q['experiment_length'] = start
        q['do_readout'] = True

        q.dc = DCbiasPulse(q)
        q.r = readoutPulse(q)

        data = runQ([q], devices)
        return processData_1q(data, q)

    axes_scans = checkAbort(gridSweep(axes), prefix=[1], func=stop_device)
    result_list = RunAllExp(runSweeper, axes_scans, dataset)
    if back:
        return result_list


@expfunc_decorator
def rabihigh(sample, measure=0, stats=1024, piamp=0.05, piLen=None, df=0*MHz,
             bias=None, zpa=None, name='rabihigh', des='', back=False):
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
    if piLen == None:
        piLen = q['piLen']

    q.power_r = power2amp(q['readout_amp']['dBm'])
    q.demod_freq = q['readout_freq'][Hz]-q['readout_mw_fc'][Hz]
    q.sb_freq = (q['f10'] - q['xy_mw_fc'])[Hz]

    # set some parameters name;
    axes = [(bias, 'bias'), (zpa, 'zpa'), (df, 'df'),
            (piamp, 'piamp'), (piLen, 'piLen')]
    deps = dependents_1q()
    kw = {'stats': stats}

    for qb in qubits:
        qb['awgs_pulse_len'] += np.max(piLen)  # add max length of hd waveforms

    # create dataset
    dataset = sweeps.prepDataset(
        sample, name+des, axes, deps, kw=kw, measure=measure)

    q_copy = q.copy()

    def runSweeper(devices, para_list):
        bias, zpa, df, piamp, piLen = para_list
        q['xy_mw_fc'] = q_copy['xy_mw_fc'] + df*Hz

        start = 0
        q.z = waveforms.square(amp=zpa, start=start, length=piLen+100e-9)
        start += 50e-9
        q.xy = [waveforms.cosine(amp=piamp, freq=q.sb_freq, start=start, length=piLen),
                waveforms.sine(amp=piamp, freq=q.sb_freq, start=start, length=piLen)]
        start += piLen + 50e-9
        q['bias'] = bias

        start += 100e-9  # additional readout gap, avoid hdawgs fall affect
        start += q['qa_start_delay'][s]  # align qa & hd start

        q['experiment_length'] = start
        q['do_readout'] = True

        q.dc = DCbiasPulse(q)
        q.r = readoutPulse(q)

        # start to run experiment
        data = runQ([q], devices)
        return processData_1q(data, q)

    axes_scans = checkAbort(gridSweep(axes), prefix=[1], func=stop_device)
    result_list = RunAllExp(runSweeper, axes_scans, dataset)
    return


@expfunc_decorator
def IQraw(sample, measure=0, stats=1024, update=False, analyze=False, reps=1,
          name='IQ raw', des='', back=True):
    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    q = qubits[measure]
    Qb = Qubits[measure]
    q.channels = dict(q['channels'])

    for qb in qubits:
        qb.power_r = power2amp(qb['readout_amp']['dBm'])
        qb.demod_freq = qb['readout_freq'][Hz]-qb['readout_mw_fc'][Hz]
    q.sb_freq = (q['f10'] - q['xy_mw_fc'])[Hz]

    # set some parameters name;
    axes = [(reps, 'reps')]
    deps = [('Is', '|0>', ''), ('Qs', '|0>', ''),
            ('Is', '|1>', ''), ('Qs', '|1>', '')]
    kw = {'stats': stats}

    # create dataset
    dataset = sweeps.prepDataset(
        sample, name+des, axes, deps, kw=kw, measure=measure)

    def runSweeper(devices, para_list):
        reps = para_list[0]
        ## with pi pulse --> |1> ##
        start = 0
        q.z = waveforms.square(
            amp=q.zpa[V], start=start, length=q.piLen[s]+100e-9)
        start += 50e-9
        q.xy = [waveforms.cosine(amp=q.piAmp, freq=q.sb_freq, start=start, length=q.piLen[s]),
                waveforms.sine(amp=q.piAmp, freq=q.sb_freq, start=start, length=q.piLen[s])]
        start += q.piLen[s] + 50e-9
        start += 100e-9
        start += q['qa_start_delay'][s]

        for _qb in qubits:
            _qb['experiment_length'] = start

        q['do_readout'] = True

        q.dc = DCbiasPulse(q)
        q.r = readoutPulse(q)

        # start to run experiment
        data1 = runQ(qubits, devices)

        ## no pi pulse --> |0> ##
        q.xy = [waveforms.square(amp=0), waveforms.square(amp=0)]

        data0 = runQ(qubits, devices)

        Is0 = np.real(data0[0])
        Qs0 = np.imag(data0[0])
        Is1 = np.real(data1[0])
        Qs1 = np.imag(data1[0])

        result = [Is0, Qs0, Is1, Qs1]
        return result

    collect, raw = True, True
    axes_scans = checkAbort(gridSweep(axes), prefix=[1], func=stop_device)

    results = RunAllExp(runSweeper, axes_scans, dataset, collect, raw)
    data = np.asarray(results[0])
    # print(data)
    if update:
        dataProcess._updateIQraw2(
            data=data, Qb=Qb, dv=None, update=update, analyze=analyze)
    return data


@expfunc_decorator
def measureFidelity(sample, rep=10, measure=0, stats=1024, update=True, analyze=False,
                    name='measureFidelity', des='', back=True):
    reps = np.arange(rep)

    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    q = qubits[measure]
    Qb = Qubits[measure]
    q.channels = dict(q['channels'])

    for qb in qubits:
        qb.demod_freq = qb['readout_freq'][Hz]-qb['readout_mw_fc'][Hz]

    q.sb_freq = (q['f10'] - q['xy_mw_fc'])[Hz]

    # set some parameters name;
    axes = [(reps, 'reps')]

    def deps_text(idxs): return ('measure |%d>, prepare (|%d>)' %
                                 (idxs[0], idxs[1]), '', '')
    deps = list(map(deps_text, itertools.product([0, 1], [0, 1])))

    kw = {'stats': stats}

    # create dataset
    dataset = sweeps.prepDataset(
        sample, name+des, axes, deps, kw=kw, measure=measure)

    def runSweeper(devices, para_list):
        ## with pi pulse --> |1> ##
        reps = para_list
        start = 0
        q.z = waveforms.square(
            amp=q.zpa[V], start=start, length=q.piLen[s]+100e-9)
        start += 50e-9

        q.xy = XYnothing(q)
        addXYgate(q, start, np.pi, 0.)

        start += q['piLen']['s'] + 50e-9
        start += 100e-9
        start += q['qa_start_delay'][s]

        for _qb in qubits:
            _qb['experiment_length'] = start
        set_qubitsDC(qubits, q['experiment_length'])
        q['do_readout'] = True
        q.r = readoutPulse(q)

        result = []
        # start to run experiment
        data1 = runQ(qubits, devices)[0]

        ## no pi pulse --> |0> ##
        q.xy = XYnothing(q)
        addXYgate(q, start, 0., 0.)

        # start to run experiment
        data0 = runQ(qubits, devices)[0]

        prob0 = tunneling([q], [data0], level=2)
        prob1 = tunneling([q], [data1], level=2)

        return [prob0[0], prob1[0], prob0[1], prob1[1]]

    axes_scans = checkAbort(gridSweep(axes), prefix=[1], func=stop_device)
    results = RunAllExp(runSweeper, axes_scans, dataset)
    if update:
        Qb['MatRead'] = np.mean(results, 0)[1:].reshape(2, 2)
    if back:
        return results


@expfunc_decorator
def T1_visibility(sample, measure=0, stats=1024, delay=0.8*us,
                  zpa=None, bias=None,
                  name='T1_visibility', des='', back=False):
    """ sample: select experimental parameter from registry;
        stats: Number of Samples for one sweep point;
    """
    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    q = qubits[measure]
    for qb in qubits:
        qb.channels = dict(qb['channels'])

    if bias == None:
        bias = q['bias']
    if zpa == None:
        zpa = q['zpa']
    q.power_r = power2amp(q['readout_amp']['dBm'])
    q.demod_freq = q['readout_freq'][Hz]-q['readout_mw_fc'][Hz]
    q.sb_freq = (q['f10'] - q['xy_mw_fc'])[Hz]

    for qb in qubits:
        qb['awgs_pulse_len'] += np.max(delay)  # add max length of hd waveforms

    # set some parameters name;
    axes = [(bias, 'bias'), (zpa, 'zpa'), (delay, 'delay')]
    deps = [('Amplitude', '1', 'a.u.'),
            ('Phase', '1', 'rad'),
            ('prob with pi pulse', '|1>', ''),
            ('Amplitude', '0', 'a.u.'),
            ('Phase', '0', 'rad'),
            ('prob without pi pulse', '|1>', '')]

    kw = {'stats': stats}

    # create dataset
    dataset = sweeps.prepDataset(
        sample, name+des, axes, deps, kw=kw, measure=measure)

    def runSweeper(devices, para_list):
        bias, zpa, delay = para_list
        # ## set device parameter

        ### ----- with pi pulse ----- ###
        start = 0
        q['bias'] = bias

        q.z = waveforms.square(amp=zpa, start=start,
                               length=delay+q.piLen[s]+100e-9)
        start += 10e-9

        q.xy = XYnothing(q)
        addXYgate(q, start, np.pi, 0.)

        start += q.piLen['s'] + delay
        start += q['qa_start_delay']['s']

        q['experiment_length'] = start
        set_qubitsDC(qubits, q['experiment_length'])
        q['do_readout'] = True
        q.r = readoutPulse(q)

        # start to run experiment
        data1 = runQ([q], devices)
        # analyze data and return
        _d_ = data1[0]
        # unit: dB; only relative strength;
        amp1 = np.mean(np.abs(_d_))/q.power_r
        phase1 = np.mean(np.angle(_d_))
        prob1 = tunneling([q], [_d_], level=2)

        ### ----- without pi pulse ----- ###
        q.xy = XYnothing(q)
        # start to run experiment
        data0 = runQ([q], devices)
        # analyze data and return
        for _d_ in data0:
            # unit: dB; only relative strength;
            amp0 = np.abs(np.mean(_d_))/q.power_r
            phase0 = np.angle(np.mean(_d_))
            prob0 = tunneling([q], [_d_], level=2)

        # multiply channel should unfold to a list for return result
        result = [amp1, phase1, prob1[1], amp0, phase0, prob0[1]]
        return result

    axes_scans = checkAbort(gridSweep(axes), prefix=[1], func=stop_device)
    result_list = RunAllExp(runSweeper, axes_scans, dataset)
    if back:
        return result_list


@expfunc_decorator
def ramsey(sample, measure=0, stats=1024, delay=ar[0:10:0.4, us],
           repetition=1, df=0*MHz, fringeFreq=10*MHz, PHASE=0,
           name='ramsey', des='', back=False):
    """ sample: select experimental parameter from registry;
        stats: Number of Samples for one sweep point;
    """
    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    q = qubits[measure]
    q.channels = dict(q['channels'])

    q.power_r = power2amp(q['readout_amp']['dBm'])
    q.demod_freq = q['readout_freq'][Hz]-q['readout_mw_fc'][Hz]
    q.sb_freq = (q['f10'] - q['xy_mw_fc'])[Hz]
    q.awgs_pulse_len += np.max(delay)  # add max length of hd waveforms

    # set some parameters name;
    axes = [(repetition, 'repetition'), (delay, 'delay'), (df, 'df'),
            (fringeFreq, 'fringeFreq'), (PHASE, 'PHASE')]
    deps = [('Amplitude', 's21 for', 'a.u.'), ('Phase', 's21 for', 'rad'),
            ('I', '', ''), ('Q', '', ''), ('prob |1>', '', '')]

    kw = {'stats': stats,
          'fringeFreq': fringeFreq}

    # create dataset
    dataset = sweeps.prepDataset(
        sample, name+des, axes, deps, kw=kw, measure=measure)

    q_copy = q.copy()

    def runSweeper(devices, para_list):
        repetition, delay, df, fringeFreq, PHASE = para_list
        # set device parameter
        q['xy_mw_fc'] = q_copy['xy_mw_fc'] + df*Hz
        ### ----- begin waveform ----- ###
        start = 0
        q.z = waveforms.square(
            amp=q.zpa[V], start=start, length=delay+2*q.piLen[s]+100e-9)
        start += 50e-9
        q.xy = XYnothing(q)
        addXYgate(q, start, theta=np.pi/2., phi=0. +
                  start*(q['f10']['Hz'])*2.*np.pi)

        start += delay + q.piLen['s']

        addXYgate(q, start, theta=np.pi/2., phi=PHASE+fringeFreq *
                  delay*2.*np.pi+start*(q['f10']['Hz'])*2.*np.pi)

        start += q.piLen[s] + 50e-9
        start += 100e-9 + q['qa_start_delay'][s]

        q['experiment_length'] = start
        set_qubitsDC(qubits, q['experiment_length'])
        q['do_readout'] = True
        q.r = readoutPulse(q)

        # start to run experiment
        data = runQ([q], devices)
        # analyze data and return
        _d_ = data[0]
        # unit: dB; only relative strength;
        amp = np.abs(np.mean(_d_))/q.power_r
        phase = np.angle(np.mean(_d_))
        Iv = np.mean(np.real(_d_))
        Qv = np.mean(np.imag(_d_))
        prob = tunneling([q], [_d_], level=2)

        # multiply channel should unfold to a list for return result
        result = [amp, phase, Iv, Qv, prob[1]]
        return result

    axes_scans = checkAbort(gridSweep(axes), prefix=[1], func=stop_device)
    result_list = RunAllExp(runSweeper, axes_scans, dataset)
    if back:
        return result_list, q


@expfunc_decorator
def s21_dispersiveShift(sample, measure=0, stats=1024, freq=ar[6.4:6.5:0.02, GHz], delay=0*ns,
                        mw_power=None, bias=None, power=None, sb_freq=None,
                        name='s21_disperShift', des='', back=False):
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
    q.awgs_pulse_len += np.max(delay)  # add max length of hd waveforms
    q.demod_freq = q['readout_freq'][Hz]-q['readout_mw_fc'][Hz]

    # set some parameters name;
    axes = [(freq, 'freq'), (bias, 'bias'), (power, 'power'), (sb_freq, 'sb_freq'), (mw_power, 'mw_power'),
            (delay, 'delay')]
    deps = [('Amplitude|0>', 'S11 for %s' % q.__name__, ''),
            ('Phase|0>', 'S11 for %s' % q.__name__, rad)]
    deps.append(('I|0>', '', ''))
    deps.append(('Q|0>', '', ''))

    deps.append(('Amplitude|1>', 'S11 for %s' % q.__name__, rad))
    deps.append(('Phase|1>', 'S11 for %s' % q.__name__, rad))
    deps.append(('I|1>', '', ''))
    deps.append(('Q|1>', '', ''))
    deps.append(('SNR', '', ''))

    kw = {'stats': stats}

    # create dataset
    dataset = sweeps.prepDataset(
        sample, name+des, axes, deps, kw=kw, measure=measure)

    def runSweeper(devices, para_list):
        freq, bias, power, sb_freq, mw_power, delay = para_list
        q['readout_amp'] = power*dBm
        q.power_r = power2amp(power)

        # set microwave source device
        q['readout_mw_fc'] = (freq - q['demod_freq'])*Hz

        # write waveforms
        ## with pi pulse --> |1> ##
        q.xy_sb_freq = (q['f10'] - q['xy_mw_fc'])[Hz]
        start = 0
        q.z = waveforms.square(
            amp=q.zpa[V], start=start, length=q.piLen[s]+100e-9)
        start += 50e-9
        q.xy = [waveforms.cosine(amp=q.piAmp, freq=q.xy_sb_freq, start=start, length=q.piLen[s]),
                waveforms.sine(amp=q.piAmp, freq=q.xy_sb_freq, start=start, length=q.piLen[s])]
        start += q.piLen[s] + 50e-9
        q['bias'] = bias

        start += delay
        start += 100e-9
        start += q['qa_start_delay'][s]

        q['experiment_length'] = start
        set_qubitsDC(qubits, q['experiment_length'])
        q['do_readout'] = True
        q.r = readoutPulse(q)

        # start to run experiment
        data1 = runQ([q], devices)

        ## no pi pulse --> |0> ##
        q.xy = [waveforms.square(amp=0), waveforms.square(amp=0)]

        # start to run experiment
        data0 = runQ([q], devices)

        # analyze data and return
        _d_ = data0[0]
        # unit: dB; only relative strength;
        amp0 = np.abs(np.mean(_d_))/q.power_r
        phase0 = np.angle(np.mean(_d_))
        Iv0 = np.mean(np.real(_d_))
        Qv0 = np.mean(np.imag(_d_))

        _d_ = data1[0]
        # unit: dB; only relative strength;
        amp1 = np.abs(np.mean(_d_))/q.power_r
        phase1 = np.angle(np.mean(_d_))
        Iv1 = np.mean(np.real(_d_))
        Qv1 = np.mean(np.imag(_d_))
        # multiply channel should unfold to a list for return result
        result = [amp0, phase0, Iv0, Qv0]
        result += [amp1, phase1, Iv1, Qv1]
        result += [np.abs((Iv1-Iv0)+1j*(Qv1-Qv0))]
        return result

    axes_scans = checkAbort(gridSweep(axes), prefix=[1], func=stop_device)
    result_list = RunAllExp(runSweeper, axes_scans, dataset)
    if back:
        return result_list


##################
# Multi qubits functions
##################

def gene_binary(qNum, qLevel=2):
    import itertools
    lbs = itertools.product(np.arange(qLevel), repeat=qNum)
    labels = []
    for i, lb in enumerate(lbs):
        label = ''.join(str(e) for e in lb)
        label = '0'*(qNum-len(label)) + label
        labels += [label]
    return labels


def prep_Nqbit(qubits):
    for _q in qubits:
        _q.channels = dict(_q['channels'])
        # _q.power_r = power2amp(_q['readout_amp']['dBm'])
        _q.demod_freq = _q['readout_freq'][Hz]-_q['readout_mw_fc'][Hz]
        _q.sb_freq = (_q['f10'] - _q['xy_mw_fc'])[Hz]

        # _q.xy = [waveforms.NOTHING,waveforms.NOTHING]
        # _q.z = waveforms.NOTHING
    return


def deps_Nqbitpopu(nq: int, qLevel: int = 2):
    labels = gene_binary(nq, qLevel)
    deps = []
    for label in labels:
        deps += [('|' + label + '>', 'prob', '')]
    return deps


@expfunc_decorator
def Nqubit_state(sample, reps=10, measure=[0, 1], states=[0, 0], name='Nqubit_state', des=''):
    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    reps = np.arange(reps)
    prep_Nqbit(qubits)
    axes = [(reps, 'reps')]
    labels = gene_binary(len(measure), 2)

    states_str = functools.reduce(lambda x, y: str(x)+str(y), states)
    def get_dep(x): return ('', 'measure '+str(x), '')
    deps = list(map(get_dep, labels))

    q_ref = qubits[0]
    kw = {}
    kw['states'] = states
    dataset = sweeps.prepDataset(sample, name+des, axes, deps, kw=kw)

    def runSweeper(devices, para_list):
        start = 0.
        for i, _qb in enumerate(qubits):
            _qb.xy = XYnothing(_qb)
            addXYgate(_qb, start, np.pi*states[i], 0.)

        start += max(map(lambda q: q['piLen']['s'], qubits)) + 50e-9
        for _q in qubits:
            _q.r = readoutPulse(_q)
            _q['experiment_length'] = start
            _q['do_readout'] = True

        set_qubitsDC(qubits, q_ref['experiment_length'])
        data = runQ(qubits, devices)
        prob = tunneling(qubits, data, level=2)
        return prob
    axes_scans = checkAbort(gridSweep(axes), prefix=[1], func=stop_device)
    result_list = RunAllExp(runSweeper, axes_scans, dataset)
    return


@expfunc_decorator
def qqiswap(sample, measure=0, delay=20*ns, zpa=None, name='iswap', des=''):
    """ 
        sample: select experimental parameter from registry;
        stats: Number of Samples for one sweep point;
    """
    sample, qubits, Qubits = loadQubits(sample, write_access=True)

    prep_Nqbit(qubits)

    q = qubits[measure]
    q_ref = qubits[0]
    if zpa == None:
        zpa = q['zpa']

    # set some parameters name;
    axes = [(delay, 'delay'), (zpa, 'zpa')]
    deps = deps_Nqbitpopu(nq=2, qLevel=2)

    # create dataset
    dataset = sweeps.prepDataset(sample, name, axes, deps, kw={})

    def runSweeper(devices, para_list):
        delay, zpa = para_list

        start = 0.

        q.xy = XYnothing(q)
        addXYgate(q, start, np.pi, 0.)

        start += q['piLen']['s'] + 50e-9

        q.z = waveforms.square(amp=zpa, start=start, length=q.piLen[s]+100e-9)

        start += 100e-9
        start += q['qa_start_delay'][s]

        for _q in qubits:
            _q.r = readoutPulse(_q)
            _q['experiment_length'] = start
            _q['do_readout'] = True

        set_qubitsDC(qubits, q_ref['experiment_length'])

        data = runQ(qubits, devices)
        prob = tunneling(qubits, data, level=2)
        return prob

    axes_scans = checkAbort(gridSweep(axes), prefix=[1], func=stop_device)
    result_list = RunAllExp(runSweeper, axes_scans, dataset)
    return


#### ----- dataprocess tools ----- ####

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
            center_i = q['center|'+str(i)+'>'][0] + \
                1j*q['center|'+str(i)+'>'][1]
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


def dependents_1q():
    deps = [('Amp', 's21 for', 'a.u.'), ('Phase', 's21 for', 'rad'),
            ('I', '', ''), ('Q', '', ''),
            ('pro', '|1>', '')]
    return deps


def processData_1q(data, q):
    """
    process single qubit IQ data into the daily used version
    """
    # analyze data and return
    _d_ = data[0]
    # only relative strength;
    amp = np.abs(np.mean(_d_))/power2amp(q['readout_amp']['dBm'])
    phase = np.angle(np.mean(_d_))
    Iv = np.real(np.mean(_d_))
    Qv = np.imag(np.mean(_d_))
    prob = tunneling([q], [_d_], level=2)
    # multiply channel should unfold to a list for return result
    result = [amp, phase, Iv, Qv, prob[1]]
    return result

#----------- dataprocess tools END -----------#


#------------- old code basket --------------#

#------------- code basket end --------------#
