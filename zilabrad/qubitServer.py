# -*- coding: utf-8 -*-
"""
Control flow for devices
"""

import time
import logging
import numpy as np
from matplotlib import pyplot as plt

from collections import Iterable
import re
import os
import copy

from zilabrad import pulse
from zilabrad.instrument.QubitContext import qubitContext

from zilabrad.pyle.tools import Unit2SI,Unit2num,convertUnits


from labrad.units import Value

np.set_printoptions(suppress=True)

# logger = logging.getLogger(__name__)
# logger.setLevel('WARNING')

def create_logger(name, filename):
    logger = logging.getLogger(name)
    logger.setLevel(logging.WARNING)

    filepath = os.path.join(os.getcwd(), filename+'.log')
    formatter = logging.Formatter(
        '%(asctime)s  %(name)s \n%(levelname)s: %(message)s\n')
    fileHandler = logging.FileHandler(filepath, encoding='UTF-8', mode='w')
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    return logger

logger = create_logger(__name__, __name__)






#####################
###  Basic Tools  ###
#####################


def add_settings(dev_dict,dev_command,parameter,dev_name=''):
    """ 
    Can check command in dict before set, 
    will raise Exception when different 
    parameter had been set in dict.
    """
    if dev_command not in dev_dict.keys():
        dev_dict[dev_command] = parameter
    elif dev_dict[dev_command] != parameter:
        raise Exception('%s Error parameter(%r) in %s'%(parameter,dev_command,dev_name))


def reset_qubits(qubits):
    """
    clear channels, start, phase 
    after each runSweeper()
    """
    clear_keys = ['z', 'xy', 'r']
    for q in qubits.values():
        for key in clear_keys:
            if q.get(key) is not None:
                q.pop(key)
        q['_start'] = 0.0
        q['_phase'] = 0.0

# def del_settings(dev_dict,dev_command,qContext=0):
#     """ 
#     TODO:
#         Use pop to delete command key 
#         in dict and copy this info
#         to qContext for debug.
#     """
#     if isinstance(qContext,int):
#         qContext = qubitContext()
#     # how to hold origin dict struction 
#     return


#######################
###  Sweeper Tools  ###
#######################

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
        # all_paras = paras[0]
        # swept_paras = paras[1]
        # 'None' is just the old devices arg which is not used now
        result = function(*all_paras)
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
                print(str(np.round(result, 3)))
            yield result

    ## run func and save data
    results = dataset.capture(wrapped())

    ## return result if require
    if collect:
        resultArray = np.asarray(list(results))
        return resultArray





def gridSweep(axes):
    """
    gridSweep generator yield all_paras, swept_paras
    if axes has one iterator, we can do a one-dimensional scanning
    if axes has two iterator, we can do a square grid scanning

    you can also create other generator, that is conditional for the
    result, do something like machnine-learning optimizer

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
        # TODO: different way to detect if something should be swept
        if np.iterable(param):
            for val in param:
                for all, swept in gridSweep(rest):
                    yield (val,) + all, (val,) + swept
        else:
            for all, swept in gridSweep(rest):
                yield (param,) + all, swept















######################
###  fake running  ###
######################

# from IPython.display import clear_output
# def live_plot(val):
#     clear_output(wait=True)
#     fig = plt.figure(figsize=(10,6))
#     ax = fig.add_subplot(1,1,1)
#     plt.plot(val[0],val[1])
#     plt.show()
#     time.sleep(0.2)


# def fake_runQubits(qubits,_settings):
#     qContext = qubitContext()
#     if 'isNewExpStart' not in qubits[0]:
#         qContext.refresh()
#     # devices_dict = qContext.get_Settings()
#     devices_dict = _settings
#     _CheckQubits(qubits)
#     _MakeWavesCorrect(qubits,devices_dict)
#     _MakeSequence(qubits,devices_dict)
#     _MakeDeviceSettings(qubits,devices_dict)

#     for dev_name,dev_command_dict in devices_dict.items():
#         if qContext.get_type(dev_name) == 'BasicDriver':
#             continue
#         for command_func in list(dev_command_dict.keys()):
#             parameter = dev_command_dict[command_func]
#             if command_func[0] == '_': ## Skip '_xxx' command
#                 continue
#             # delete command after setting
#             devices_dict[dev_name].pop(command_func)
#             t1 = time.time()
#             print('[%s] %s'%(dev_name,command_func))

#     return [-1]*10



###################
###  runQubits  ###
###################


def runQubits_raw(qubits,devices,measure=None,reset=True):
    """ run Qubits with directly sending waveform from q.xy/z/r
    """
    if measure is None:
        measure = list(qubits.keys())
    # qContext = qubitContext()
    # if 'isNewExpStart' not in qubits[0]:
    #     qContext.refresh()
    # devices_dict = qContext.get_Settings()
    devices_dict = devices['_taskflow']
    debug_ctrl = devices['debug']
    trig_device = devices['trigger']
    # _CheckQubits(qubits)
    # _MakeWavesCorrect(qubits,devices_dict,measure)
    _MakeSequence_raw(qubits,devices_dict,devices)
    if 'isNewExpStart' not in devices: 
        ## set them only once
        demod_dict = _MakeDemodulator(qubits,devices_dict,measure)
        devices['demod_info'] = demod_dict
        ## mark after the first runQubits()
        devices['isNewExpStart'] = False
    _SyncDevices(devices_dict,debug=debug_ctrl,trig=trig_device)
    data = _RunDevices(devices_dict,devices['demod_info'],trig=trig_device)
    if reset:
        reset_qubits(qubits)
    return data


    
def _MakeSequence_raw(qubits,devices_dict,devices,back=False):
    qContext = qubitContext()
    waveGenerator = qContext.get_server('waveGenerator')
    waves_Func = {}  ## temporary dict for qubits pulse, avoid overwriting waveforms
    trigger_start = -Unit2SI(devices['waveGenerator']['bias_rise'])
    trigger_end = Unit2SI(devices['waveGenerator']['awg_pulse_length']
                        +devices['waveGenerator']['bias_fall'])
    for qb in qubits.values():
        for ch_type in ['xy','z','readout']:
            ## info -> {ch_type:'dev_name-ch_path'}
            dev_name,ch_path = qb['channels'][ch_type].split('-')
            ## initial device dict
            if dev_name not in waves_Func.keys():
                waves_Func[dev_name] = {}
                
            path_type,path_num = re.findall(r'[A-Za-z]+|\d+', ch_path)
            wave_factor = 1
            ## make sure 'awg'+idx in dict.keys(), fit form in dev.send_waveform()
            if (path_type == 'ch') or (path_type =='wave') or (path_type =='channel'):
                ch_path = 'awg'+str((int(path_num)-1)//2)
                if int((int(path_num)-1)%2) == 1:
                    wave_factor = 1j
                    
            ## check qb.channels information
            if 'awg' not in ch_path:
                raise Exception('Error channel path: %r'%ch_path)
            # if ch_type not in ['xy','z','readout']:
            #     raise Exception('Error channel type: %r'%ch_type)
                
            ## initial device's waveforms Envelope
            if ch_path not in waves_Func[dev_name]:
                if (ch_type == 'z') and ('z' in qb.keys()):
                    waves_Func[dev_name][ch_path] = wave_factor*qb['z']
                if (ch_type == 'xy') and ('xy' in qb.keys()):
                    waves_Func[dev_name][ch_path] = wave_factor*qb['xy']
                if (ch_type == 'readout') and ('r' in qb.keys()):
                    waves_Func[dev_name][ch_path] = pulse.NOTHING
            else:
                if (ch_type == 'z') and ('z' in qb.keys()):
                    waves_Func[dev_name][ch_path] += wave_factor*qb['z']
                if (ch_type == 'xy') and ('xy' in qb.keys()):
                    waves_Func[dev_name][ch_path] += wave_factor*qb['xy']
                if (ch_type == 'readout') and ('r' in qb.keys()):
                    waves_Func[dev_name][ch_path] += wave_factor*qb['r']
            

    # compile waveforms Envelope to array, and send to devices_dict
    for dev_name,waves_dict in waves_Func.items():
        for ch_path,IQwaves in waves_dict.items():
            if dev_name not in devices_dict.keys():
                devices_dict[dev_name] = {}
            if '_waves' not in devices_dict[dev_name].keys():
                devices_dict[dev_name]['_waves'] = {}
            dev_type = qContext.get_type(dev_name)
            if dev_type == 'ArbitraryWaveGenerator':
                devices_dict[dev_name]['_waves'][ch_path] = [IQwaves.real,IQwaves.imag]
            if dev_type == 'QuantumAnalyzer':
                ## multiple of 40 ns for (qa.set_pulse_start)
                _length = IQwaves.start  # unit: s
                _length_40ns = ((_length*1e9) // 40)*40e-9  # unit: s
                devices_dict[dev_name]['set_pulse_start'] = _length_40ns
                devices_dict[dev_name]['_waves'][ch_path] = waveGenerator.func2array(
                    IQwaves,start=_length_40ns,end=IQwaves.end,fs=1.8e9,mode='IQ')
    if back:
        return devices_dict
 








def runQubits(qubits,devices,measure=None,reset=True):
    if measure is None:
        measure = list(qubits.keys())
    # qContext = qubitContext()
    # if 'isNewExpStart' not in qubits[0]:
    #     qContext.refresh()
    # devices_dict = qContext.get_Settings()
    devices_dict = devices['_taskflow']
    debug_ctrl = devices['debug']
    trig_device = devices['trigger']
    # _CheckQubits(qubits)
    _MakeWavesCorrect(qubits,devices_dict,measure)
    _MakeSequence(qubits,devices_dict,devices)
    if 'isNewExpStart' not in devices: 
        ## set them only once
        demod_dict = _MakeDemodulator(qubits,devices_dict,measure)
        devices['demod_info'] = demod_dict
        ## mark after the first runQubits()
        devices['isNewExpStart'] = False
    _SyncDevices(devices_dict,debug=debug_ctrl,trig=trig_device)
    data = _RunDevices(devices_dict,devices['demod_info'],trig=trig_device)
    if reset:
        reset_qubits(qubits)
    return data

def _CheckQubits(qubits):

    for qb in qubits.values():
        # if  qubits[0]['stats'] != qb['stats']:
        #     raise Exception('Must set same stats in Qubits')
        if abs(Unit2SI(qubits[0]['demod_start'])-Unit2SI(qb['demod_start'])) > 0.01e-9:
            raise Exception('Must set same qa_demod_start in Qubits')
                                    
        # if abs(Unit2SI(qubits[0]['qa_demod_length'])-Unit2SI(qb['qa_demod_length'])) > 0.01e-9:
        #     raise Exception('Must set same qa_demod_length in Qubits')

        # if abs(Unit2SI(qubits[0]['bias_start'])-Unit2SI(qb['bias_start'])) > 0.01e-9:
        #     raise Exception('Must set same bias_start in Qubits')

def _MakeWavesCorrect(qubits,devices_dict,measure,sequence_delay=True):
    if sequence_delay:
        for qb in qubits.values():
            for ch in ['xy','z','r']: ## for xy/z/r pulse
                if ch in  qb.keys():
                    delay = Unit2SI(qb['correct']['sequence_delay'][ch])
                    qb[ch] = pulse.time_shift(qb[ch],delay=delay)
            if 'demod_start' in qb.keys():  ## for readout demodulator windows
                delay = Unit2SI(qb['correct']['sequence_delay']['demod'])
                qb['demod_start'] =  Unit2SI(qb['demod_start']) + delay
    
    ''' owing to fixed length between trigger and demodulator start
        (qb.qa_demod_start) must multiple of 40 ns 
    '''
    _length = Unit2SI(qubits[measure[0]]['demod_start']) ## unit: s
    _length_40ns = ((_length*1e9+39.9) // 40)*40e-9  # unit: s
    trigger_shift = _length_40ns - _length # unit: s
    for qb in qubits.values():
        if 'xy' in qb.keys(): ## for xy pulse
            qb['xy'] = pulse.time_shift(qb['xy'],delay=trigger_shift)
        if 'z' in qb.keys():  ## for z pulse
            qb['z'] = pulse.time_shift(qb['z'],delay=trigger_shift)
        if 'r' in qb.keys():  ## for readin pulse
            qb['r'] = pulse.time_shift(qb['r'],delay=trigger_shift)
        if 'demod_start' in qb.keys():  ## for readout demodulator windows
            qb['demod_start'] =  Unit2SI(qb['demod_start']) + trigger_shift
    
def _MakeSequence(qubits,devices_dict,devices,back=False):
    qContext = qubitContext()
    waveGenerator = qContext.get_server('waveGenerator')
    waves_Func = {}  ## temporary dict for qubits pulse, avoid overwriting waveforms
    trigger_start = -Unit2SI(devices['waveGenerator']['bias_rise'])
    trigger_end = Unit2SI(devices['waveGenerator']['awg_pulse_length']
                        +devices['waveGenerator']['bias_fall'])
    for qb in qubits.values():
        for qb_key in ['xy','z','r']:
            if qb_key in qb.keys():
                if qb_key == 'r':
                    ch_type = 'readout'
                else:
                    ch_type = qb_key
                ## info -> {ch_type:'dev_name-ch_path'}
                dev_name,ch_path = qb['channels'][ch_type].split('-')
                ## initial device dict
                if dev_name not in waves_Func.keys():
                    waves_Func[dev_name] = {}
                    
                path_type,path_num = re.findall(r'[A-Za-z]+|\d+', ch_path)
                wave_factor = 1
                ## make sure 'awg'+idx in dict.keys(), fit form in dev.send_waveform()
                if (path_type == 'ch') or (path_type =='wave') or (path_type =='channel'):
                    ch_path = 'awg'+str((int(path_num)-1)//2)
                    if int((int(path_num)-1)%2) == 1:
                        wave_factor = 1j
                        
                ## check qb.channels information
                if 'awg' not in ch_path:
                    raise Exception('Error channel path: %r'%ch_path)
                # if ch_type not in ['xy','z','readout']:
                #     raise Exception('Error channel type: %r'%ch_type)
                    
                ## initial device's waveforms Envelope
                if ch_path not in waves_Func[dev_name]:
                    waves_Func[dev_name][ch_path] = pulse.NOTHING
                waves_Func[dev_name][ch_path] += wave_factor*qb[qb_key]

    # compile waveforms Envelope to array, and send to devices_dict
    for dev_name,waves_dict in waves_Func.items():
        for ch_path,IQwaves in waves_dict.items():
            if dev_name not in devices_dict.keys():
                devices_dict[dev_name] = {}
            if '_waves' not in devices_dict[dev_name].keys():
                devices_dict[dev_name]['_waves'] = {}
            dev_type = qContext.get_type(dev_name)
            if dev_type == 'ArbitraryWaveGenerator':
                devices_dict[dev_name]['_waves'][ch_path] = waveGenerator.func2array(
                    IQwaves,start=trigger_start,end=trigger_end,fs=2.4e9,mode='IQ')
            if dev_type == 'QuantumAnalyzer':
                ## multiple of 40 ns for (qa.set_pulse_start)
                _length = IQwaves.start  # unit: s
                _length_40ns = ((_length*1e9) // 40)*40e-9  # unit: s
                devices_dict[dev_name]['set_pulse_start'] = _length_40ns
                devices_dict[dev_name]['_waves'][ch_path] = waveGenerator.func2array(
                    IQwaves,start=_length_40ns,end=IQwaves.end,fs=1.8e9,mode='IQ')
    if back:
        return devices_dict
            


def _MakeDemodulator(qubits,devices_dict,measure):
    demod_dict = {}
    for qb in qubits.values():
        ## for qubits readout settings
        if qb.get('do_readout'):
            dev_name,ch_num = qb['channels']['readout'].split('-')

            if dev_name not in devices_dict.keys(): 
                devices_dict[dev_name] = {}
                
            if 'set_demod_start' not in devices_dict[dev_name].keys():
                wait_time = qb['demod_start']
                if 'set_pulse_start' in devices_dict[dev_name]:
                    wait_time -= devices_dict[dev_name]['set_pulse_start']
                devices_dict[dev_name]['set_demod_start'] = wait_time

            if 'set_qubit_frequency' not in devices_dict[dev_name].keys():
                devices_dict[dev_name]['set_qubit_frequency'] = []
            if qb.__name__ in measure: ## only read qubits in measure list
                devices_dict[dev_name]['set_qubit_frequency'] += [Unit2SI(qb['demod_freq'])]
                if dev_name not in demod_dict:
                    demod_dict[dev_name] = []
                demod_dict[dev_name] += [qb.__name__]
    return demod_dict


def _SyncDevices(devices_dict,trig='qa_1',debug=False):
    qContext = qubitContext()
    if debug:
        ## copy for debug -- by hwh 20210406
        qContext.devices_dict_copy = devices_dict.copy()

    for dev_name,dev_command_dict in devices_dict.items():
        if qContext.get_type(dev_name) == 'BasicDriver':
            continue
        dev = qContext.get_server(dev_name)
        ## need to distinguish device's type for settings
        ## cannot del key during dict.items(), should convert to list
        for command_func in list(dev_command_dict.keys()):
            parameter = dev_command_dict[command_func]
            if command_func[0] == '_': ## skip '_xxx' command
                continue
            ## log sync command
            logger.debug('SyncDevices: [%s] %s'%(dev_name,command_func))
            t0 = time.time()
            if parameter is 'None':
                getattr(dev,command_func)()
            elif isinstance(parameter,Iterable) and not isinstance(parameter,(Value,str)):  ## for many input
                getattr(dev,command_func)(*parameter)
            else:              
                getattr(dev,command_func)(parameter)
            # delete command after setting
            devices_dict[dev_name].pop(command_func)
            t1 = time.time()
            logger.debug('use time: %.2f ms'%((t1-t0)*1e3))
            if debug:
                print('[%s] %s ; Use time: %.2f ms'%(dev_name,command_func,(t1-t0)*1e3))

        ## for sending waveforms to devices
        if '_waves' in dev_command_dict.keys():
            waveforms_dict = dev_command_dict['_waves']
            for ch_path in list(waveforms_dict.keys()):
                waves = waveforms_dict[ch_path]
                path_type,path_num = re.findall(r'[A-Za-z]+|\d+', ch_path)
                logger.debug('SyncDevices: [%s-%s] send_waveform'%(dev_name,ch_path))
                t0 = time.time()
                dev.send_waveform(waves,int(path_num))
                ## delete wavefrom after send
                devices_dict[dev_name]['_waves'].pop(ch_path)
                ## start slave awgs running and wait trigger
                if dev_name is not trig:
                    dev.awg_run(int(path_num))
                t1 = time.time()
                logger.debug('use time: %.2f ms'%((t1-t0)*1e3))
                if debug:
                    print('[%s-%s] send waveform ; Use time: %.2f ms'%(dev_name,ch_path,(t1-t0)*1e3))
                    # print('[%s-%s] awg_run ; Use time: %.2f ms'%(dev_name,ch_path,(t1-t0)*1e3))



def _RunDevices(devices_dict,demod_dict,trig='qa_1'):
    """ 
    Start master device's trigger,
    and do data acquisition
    """
    qContext = qubitContext()
    ## str: only one master
    master_dev = qContext.get_server(trig)
    master_dev.awg_run()

    """
    get data from readout channels of measure qubits
    return dict like: {'q1':raw_data,'q2':raw_data}
    """
    raw_data = {}
    for dev_name,meas_list in demod_dict.items():
        meas_devive = qContext.get_server(dev_name)
        _raw_list = meas_devive.get_data()
        for k in range(len(meas_list)):
            raw_data[meas_list[k]] = _raw_list[k]
    return raw_data


