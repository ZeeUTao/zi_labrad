# -*- coding: utf-8 -*-
"""
qubitServer to control all instruments

Created on 2020.09.09 20:57
@author: Tao Ziyu
"""

import time
from functools import wraps
import logging
import numpy as np 

from zilabrad.instrument import waveforms
from zilabrad.instrument.zurichHelper import zurich_qa, zurich_hd
from zilabrad.instrument.QubitDict import loadQubits,update_session,loadInfo
from zilabrad.pyle.pipeline import pmap

import labrad
from labrad.units import Unit,Value
_unitSpace = ('V','mV','us','ns','s','GHz','MHz','kHz','Hz','dBm','rad','None')
V, mV, us, ns,s, GHz, MHz,kHz,Hz, dBm, rad,_l  = [Unit(s) for s in _unitSpace]



np.set_printoptions(suppress=True)
_noisy_printData = True


_type2Regkey = {
'qa':'ziQA_id',
'hd':'ziHD_id',
'mw':'microwave_source'
}
"""
dict for device type to the name of key in Registry
"""

_server_class = {
'qa':zurich_qa,
'hd':zurich_hd,
}
"""
dict for device type to the server class
"""


def get_deviceMap(_type: str):
    """
    get a device Mapping dictionary
    choose a simpler name '1', '2', '3', not 'dev8334'
    Example:
        [('1', 'dev8334'),('2', 'dev8335')]
    """
    if _type not in _type2Regkey:
        raise TypeError("No such device type %s"%(_type))
    dev = loadInfo(paths=['Servers','devices'])
    deviceMap = dict(dev[_type2Regkey[_type]])
    return deviceMap
    
def get_microwaveServer():
    """
    usually return anritsu_server
    """
    dev = loadInfo(paths=['Servers','devices'])
    return str(dev['microwave_server'])
    
    
def sortDevice(_type: str):
    """
    Args: 
        _type: device type, 'qa', 'hd'
    Returns:
        a dictionary, for example {'1',object}, object is an instance of device server (class)
        Do not worry about the instance of the same device is recreated, which is set into a conditional singleton.
    """
    
    
    dev = loadInfo(paths=['Servers','devices'])
    deviceMap = get_deviceMap(_type)
    
    deviceDict = {}
    
    for _id in deviceMap:
        if _type in ['qa','hd']:
            server = _server_class[_type]
            # if use labone zurich instrument
            deviceDict = server(_id,device_id=deviceMap[_id],labone_ip=dev['labone_ip'])
        else:
            raise TypeError("No such device type %s"%(_type))
            
    return deviceDict



def check_device():
    """
    Make sure all device in work before runQ
    """
    # cxn = labrad.connect()
    
    # server = cxn[get_microwaveServer()]
    # deviceMap = get_deviceMap('mw')    
    # for i,key in enumerate(deviceMap):
        # server.select_device(deviceMap[key])
        # server.output(True)
    return



def stop_device():
    """  close all device; 
    """
    deviceDict_qa = sortDevice('qa')
    for key in deviceDict_qa.keys():
        server = deviceDict_qa[key]
        server.stop_subscribe()
    
    deviceDict_hd = sortDevice('hd')
    for key in deviceDict_hd.keys():
        server = deviceDict_hd[key]
        server.awg_close_all()
        
    cxn = labrad.connect()
    
    server = cxn[get_microwaveServer()]
    deviceMap = get_deviceMap('mw')    
    for i,key in enumerate(deviceMap):
        server.select_device(deviceMap[key])
        server.output(False)
    return
    
    
def dataset_create(dataset):
    """Create the dataset. 
    see 
    dataset = sweeps.prepDataset(*args)
    dv = labrad.connect().dataVault
    dataVault script is in "/server/py3_data_vault"
    """
    cxn = labrad.connect()
    dv = cxn.data_vault
    
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




def _mpAwg_init(qubits:list):
    """
    prepare and Returns waveforms
    Args:
        qubits (list) -> [q (dict)]: contains value as parameter
        qa (object): zurich_qa instance
        hd (object): zurich_hd instance
    
    Returns:
        w_qa (object): waveform instance for qa
        w_hd (object): waveform instance for hd

    TODO: better way to generate w_qa, w_hd
    try to create features in the exsiting class but not create a new function
    """
    qa = sortDevice('qa')
    qa = qa['1']
    
    hd = sortDevice('hd')
    hd = hd['1']
    
    q = qubits[0] ## the first qubit as master

    hd.pulse_length_s = 0 ## add hdawgs length with unit[s]

    qa.result_samples = qubits[0]['stats']  ## int: sample number for one sweep point
    qa.set_adc_trig_delay(q['bias_start']+hd.pulse_length_s*s)
    qa.set_readout_delay(q['readout_delay'])
    qa.set_pulse_length(q['readout_len'])

    f_read = []
    for qb in qubits:
        if qb['do_readout']: ##in _q.keys():
            f_read += [qb.demod_freq]
    if len(f_read) == 0:
        raise Exception('Must set one readout frequency at least')
    qa.set_qubit_frequency(f_read) ##

    ## initialize waveforms and building 
    # qa.update_wave_length()
    # hd.update_wave_length()
    ### ----- finish ----- ###
    return 
    
    
def RunAllExperiment(function,iterable,dataset,
                     collect: bool = True,
                     raw: bool = False,
                     pipesize: int = 10):
    """ Define an abstract loop to iterate a funtion for iterable

    Example:
    prepare parameters: scanning axes, dependences and other parameters
    start experiment with special sequences according to the parameters
    
    Args:
        function: give special sequences, parameters, and start exp_devices, that are to be called by run()
        iterable: iterated over to produce values that are fed to function as parameters.
        dataset: zilabrad.pyle.datasaver.Dataset
        collect: if True, collect the result into an array and return it; else, return an empty list
        raw: discard swept_paras if raw == True
    """
    def run(paras):
        # pass in all_paras to the function
        all_paras = [Unit2SI(a) for a in paras[0]]
        swept_paras = [Unit2num(a) for a in paras[1]]
        result = function(None,all_paras) # 'None' is just the old devices arg which is not used now 
        if raw:
            result_raws = np.asarray(result)
            return result_raws.T
        else:
            result = np.hstack([swept_paras,result])
            return result

    def wrapped():
        for paras in iterable:
            result = run(paras)
            if _noisy_printData == True:
                print(
                    str(np.round(result,3))
                    )
            yield result
            
    
    # iter = pmap(wrapped, iterable, size=pipesize)
    results = dataset.capture(wrapped())
        
    result_list = []
    for result in results:
        result_list.append(result)
    
    if collect:
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
    

def set_microwaveSource(freqList,powerList):
    """set frequency and power for microwaveSource devices
    """
    print(freqList[1]['MHz'],powerList[1]['dBm'])
    deviceMap = get_deviceMap('mw')   
    cxn = labrad.connect()
    server = cxn[get_microwaveServer()]
    deviceMap = get_deviceMap('mw')    
    for i,key in enumerate(deviceMap):
        server.select_device(deviceMap[key])
        server.output(True)
        
        server.frequency(freqList[i]['MHz'])
        server.amplitude(powerList[i]['dBm'])
    return
        


def makeSequence_readout(qubits):
    """
    waveServer: zilabrad.instrument.waveforms
    This version only, consider one zurich_qa device
    FS: sampling rates
    """
    qa = sortDevice('qa')
    qa = qa['1']
    FS = qa.FS
    waveServer = waveforms.waveServer(device_id='0')
    
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


def makeSequence_AWG(qubits):
    """
    waveServer: zilabrad.instrument.waveforms
    FS: sampling rates
    """
    hd = sortDevice('hd')
    hd = hd['1']
    
    FS = hd.FS

    
    wave_AWG = []
    waveServer = waveforms.waveServer(device_id='0')
    
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
            
    return wave_AWG

def setupDevices(qubits):
    qa = sortDevice('qa')
    qa = qa['1']
    
    hd = sortDevice('hd')
    hd = hd['1']

    q_ref = qubits[0]
    
    # initialization
    if hd.pulse_length_s != q_ref['experiment_length']:
        hd.pulse_length_s = q_ref['experiment_length']
        qa.set_adc_trig_delay(q_ref['bias_start'][s]+hd.pulse_length_s)
    
    if 'do_init' not in qubits[0].keys():
        _mpAwg_init(qubits)
        qubits[0]['do_init']=True
        logging.info('do_init')

def runDevices(qubits,wave_AWG,wave_readout):
    qa = sortDevice('qa')
    qa = qa['1']
    
    hd = sortDevice('hd')
    hd = hd['1']


    qubit_ports = getQubits_awgPort(qubits)
    hd.send_waveform(waveform=wave_AWG, ports=qubit_ports)
    qa.send_waveform(waveform=wave_readout)

    ## start to run experiment
    hd.awg_open()
    qa.awg_open()
    data = qa.get_data()
    return data
        
        
def runQubits(qubits,exp_devices = None):
    """ generally for running multiqubits

    Args:
        qubits (list): a list of dictionary
        _runQ_servers (list/tuple): instances used to control device
    
    Time Cost:
        0.3 s : prepare waveform in local PC
        0.2 s : setupDevices
        0.8 s : runDevices and get data
    TODO:
        (1) check _is_runfirst=True? Need run '_mpAwg_init' at first running;
        (2) clear q.xy/z/dc and their array after send();
    """

    
    # prepare wave packets
    wave_AWG = makeSequence_AWG(qubits)
    wave_readout = makeSequence_readout(qubits)

    q_ref = qubits[0]
    
    ## Now it is only two microwave sources, in the future, it should be modified
    set_microwaveSource(freqList = [q_ref['xy_mw_fc'],q_ref['readout_mw_fc']],
                        powerList = [q_ref['xy_mw_power'],q_ref['readout_mw_power']])
    
    setupDevices(qubits)    

    ## run AWG and reaout devices and get data
    data = runDevices(qubits,wave_AWG,wave_readout)
    return data

