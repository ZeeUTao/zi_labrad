# -*- coding: utf-8 -*-
"""
Control flow for devices
"""

import time
from functools import wraps
import logging
import numpy as np
import gc

from zilabrad.instrument import waveforms
from zilabrad.instrument.zurichHelper import zurich_qa, zurich_hd
from zilabrad.instrument.QubitContext import loadQubits
from zilabrad.instrument.QubitContext import qubitContext

from labrad.units import Unit, Value

np.set_printoptions(suppress=True)

logger = logging.getLogger(__name__)
logger.setLevel('WARNING')


def stop_device():
    """  close all device;
    """
    qContext = qubitContext()
    zurich_devices = qContext.get_servers_group(type='zurich').values()
    for server in zurich_devices:
        server.awg_close()

    uwave_source = qContext.get_server(
        type='microwave_source', name=None)
    uwave_source.stop_all()
    return


def Unit2SI(a):
    if type(a) is not Value:
        return a
    elif a.unit in ['GHz', 'MHz']:
        return a['Hz']
    elif a.unit in ['ns', 'us']:
        return a['s']
    else:
        return a[a.unit]


def Unit2num(a):
    if type(a) is not Value:
        return a
    else:
        return a[a.unit]


def RunAllExperiment(
        function, iterable, dataset,
        collect=True, raw=False, noisy=False
        ):
    """ Define an abstract loop to iterate a funtion for iterable

    Example:
    prepare parameters: scanning axes, dependences and other parameters
    start experiment with special sequences according to the parameters

    Args:
        function: give special sequences, parameters
        iterable: iterated over to produce values that are fed to function
        as parameters.
        dataset: zilabrad.pyle.datasaver.Dataset, object for data saving
        collect: if True, collect the result into an array and return it;
        else, return an empty list
        raw: discard swept_paras if raw == True
    """
    def run(paras):
        # pass in all_paras to the function
        all_paras = [Unit2SI(a) for a in paras[0]]
        swept_paras = [Unit2num(a) for a in paras[1]]
        # 'None' is just the old devices arg which is not used now
        result = function(None, all_paras)
        if raw:
            result_raws = np.asarray(result)
            return result_raws.T
        else:
            result = np.hstack([swept_paras, result])
            return result

    def wrapped():
        for paras in iterable:
            result = run(paras)
            if noisy is True:
                print(
                    str(np.round(result, 3))
                )
            yield result

    gc_var = gc.collect()
    print(f"garbage collect {gc_var}")

    qContext = qubitContext()

    results = dataset.capture(wrapped())
    resultArray = np.asarray(list(results))
    if collect:
        return resultArray


def set_microwaveSource(freqList, powerList):
    """set frequency and power for microwaveSource devices
    """
    qContext = qubitContext()
    server = qContext.get_server(
        type='microwave_source', name=None)
    IPdict = qContext.IPdict_microwave

    for i, key in enumerate(IPdict):
        server.select_device(IPdict[key])
        server.output(True)
        server.frequency(freqList[i]['MHz'])
        server.amplitude(powerList[i]['dBm'])
    return


def makeSequence_readout(qubits, FS=1.8e9):
    """
    waveServer: zilabrad.instrument.waveforms
    We assume all zurich_qa devices have the
    same sampling rate.
    FS: sampling rates
    """
    waveServer = waveforms.waveServer()

    wave_readout_func = [waveforms.NOTHING, waveforms.NOTHING]
    for q in qubits:
        if q.get('do_readout'):
            if 'r' in q.keys():
                wave_readout_func[0] += q.r[0]
                wave_readout_func[1] += q.r[1]
            else:
                print('Error! No readout pulse!')

    q_ref = qubits[0]
    start = 0.
    end = q_ref['readout_len']['s']

    waveServer.set_tlist(start=start, end=end, fs=FS)
    wave_readout = [waveServer.func2array(
        wave_readout_func[i], start, end) for i in [0, 1]]
    return wave_readout


def makeSequence_AWG(qubits, FS=2.4e9):
    """
    waveServer: zilabrad.instrument.waveforms
    FS: sampling rates
    """
    wave_AWG = []
    waveServer = waveforms.waveServer()

    # This version consider all of the ports of AWG
    # require the same sampling points

    # ----bias_start previously -----  XY,Z, pulse quantum getes...
    # ------ readout ----- bias_end ---
    q_ref = qubits[0]
    start = -Unit2SI(q_ref['bias_start'])
    end = Unit2SI(q_ref['bias_end']) + \
        Unit2SI(q_ref['awgs_pulse_len']) + Unit2SI(q_ref['readout_len'])

    # this is just set parameters, not really generate a list, which is slow
    waveServer.set_tlist(start, end, fs=FS)

    for q in qubits:
        # the order must be 'dc,xy,z' ! match the order in QubitContext

        # line [DC]
        if 'dc' in q.keys():
            if q.get('dc') is None:
                wave_AWG += [None]
            else:
                wave_AWG += [waveServer.func2array(q.dc, start, end, FS)]

        # line [xy] I,Q
        if 'xy' in q.keys():
            wave_AWG += [waveServer.func2array((q.xy)[i], start, end, FS)
                         for i in [0, 1]]

        # line [z]
        if 'z' in q.keys():
            wave_AWG += [waveServer.func2array(q.z, start, end, FS)]

    return wave_AWG


def setup_wiring(qContext):
    '''
    QA mode:
        2 --> Rotation;
        7 --> Integration;
    HD mode:
        0 --> 4*2 awgs;
        1 --> 2*4 awgs;
        2 --> 1*8 awgs;
    '''
    _wiring_mode = qContext.wiring
    for name, mode in _wiring_mode.items():
        if 'qa' in name:
            qContext.servers_qa[name].set_qaSource_mode(mode)
        elif 'hd' in name:
            qContext.servers_hd[name].awg_grouping(mode)
        else:
            print('Unknown name (%r) with mode(%r)' % (name, mode))


def setupDevices(qubits):
    q_ref = qubits[0]
    # only run once in the whole experimental loop
    qContext = qubitContext()
    # qas = qContext.get_servers_group('qa')
    qa = qContext.get_server('qa', 'qa_1')

    if 'isNewExpStart' not in q_ref:
        print('isNewExpStart, setupDevices')
        qContext.refresh()
        setup_wiring(qContext)
        # int: sample number for one sweep point
        qa.set_result_samples(q_ref['stats'])

        # only related with wiring and devices, delay between QA
        # signal output and demodulation
        qa.set_readout_delay(q_ref['readout_delay'])

        # set qa pulse length in AWGs, and set same length for demodulate.
        qa.set_pulse_length(q_ref['readout_len'])

        # delay between zurich HD and QA
        qa.set_adc_trig_delay(
            q_ref['bias_start']['s']+q_ref['experiment_length'])

        # set demodulate frequency for qubits if you need readout the qubit
        f_read = []
        for qb in qubits:
            if qb.get('do_readout'):  # in _q.keys():
                f_read += [qb.demod_freq]
        if len(f_read) == 0:
            raise Exception('Must set one readout frequency at least')
        qa.set_qubit_frequency(f_read)

        q_ref['isNewExpStart'] = False  # actually it can be arbitrary value

    else:
        # delay between zurich HD and QA
        # for example: in T1 measurement
        qa.set_adc_trig_delay(
            q_ref['bias_start']['s']+q_ref['experiment_length'])
    return


def runDevices(qubits, wave_AWG, wave_readout):
    qContext = qubitContext()
    # qas = qContext.get_servers_group('qa')
    qa = qContext.get_server('qa', 'qa_1')
    # send data packet to multiply devices
    qa.send_waveform(waveform=wave_readout)

    hds = qContext.get_servers_group('hd')
    qubits_port = qContext.getPorts(qubits)
    awg_waves = AWG_wave_dict(qubits_port, wave_AWG)

    for (dev_name, awg_index), wave in awg_waves.items():
        hd = hds[dev_name]
        hd.send_waveform(
            waveform=wave, awg_index=awg_index)
        hd.awg_open(awgs_index=[awg_index])

    qa.awg_open()  # download experimental data
    _data = qa.get_data()

    if qa.source == 7:  # single channels
        return _data
    else:  # double channels
        ks = range(int(len(_data)/2))
        def get_doubleChannel(k): return _data[2*k]+1j*_data[2*k+1]
        data_doubleChannel = list(map(get_doubleChannel, ks))
        return data_doubleChannel


def AWG_wave_dict(devices_info, waves):
    """ Combine waveform sequence and device info (name, port)
    as dictionary.
    Args:
        devices_info (list): [(dev_name, channel)], channel is in [1,2...8]
        for example, [('hd_1', 3), ('hd_1', 4)]
    Return:
        port_dict = {
            (dev_name, awg_index): [wave1, wave2]
            }
    """
    port_dict = {}
    for wave in waves:
        if wave is not None:
            wave_len = len(wave)
            break

    for k, info in enumerate(devices_info):
        if info is None:
            continue
        dev_name, channel = info
        awg_index = (channel-1) // 2
        _key = (dev_name, awg_index)
        # initiate
        if port_dict.get(_key) is None:
            port_dict[_key] = [None, None]

        # add waves
        wave_order = (channel-1) % 2
        wave = waves[k]
        # wave_order == 0 or 1
        port_dict[_key][wave_order] = wave

    # fill zero array if wave_list still has None
    for wave_list in port_dict.values():
        for i in range(len(wave_list)):
            if wave_list[i] is None:
                wave_list[i] = np.zeros(wave_len)

    return port_dict


def runQubits(qubits, exp_devices=None):
    """ generally for running multiqubits
    Args:
        qubits (list): a list of dictionary
    """
    qContext = qubitContext()

    wave_readout = makeSequence_readout(qubits, FS=qContext.ADC_FS)
    wave_AWG = makeSequence_AWG(qubits, FS=qContext.DAC_FS)

    q_ref = qubits[0]
    # Now it is only two microwave sources, more should be covered
    # in the future
    set_microwaveSource(
        freqList=[q_ref['readout_mw_fc'], q_ref['xy_mw_fc']],
        powerList=[q_ref['readout_mw_power'], q_ref['xy_mw_power']])

    setupDevices(qubits)
    data = runDevices(qubits, wave_AWG, wave_readout)

    return data
