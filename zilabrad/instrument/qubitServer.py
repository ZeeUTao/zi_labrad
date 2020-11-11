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
from zilabrad.instrument.QubitContext import loadQubits
from zilabrad.instrument.QubitContext import qubitContext
from zilabrad.pyle.pipeline import pmap


import labrad
from labrad.units import Unit,Value
_unitSpace = ('V','mV','us','ns','s','GHz','MHz','kHz','Hz','dBm','rad','None')
V, mV, us, ns,s, GHz, MHz,kHz,Hz, dBm, rad,_l  = [Unit(s) for s in _unitSpace]



np.set_printoptions(suppress=True)
_noisy_printData = True




def check_device():
    """
    Make sure all device in work before runQ
    """
    qContext = qubitContext()
    server = qContext.servers_microwave
    IPdict = qContext.IPdict_microwave

    for key,value in IPdict.items():
        server.select_device(value)
        server.output(True)
    return



def stop_device():
    """  close all device; 
    """
    qContext = qubitContext()
    serverDict = qContext.servers_qa 
    for key,server in serverDict.items():
        server.awg_close()
    
    serverDict = qContext.servers_hd
    for key,server in serverDict.items():
        server.awg_close()
        
    server = qContext.servers_microwave
    IPdict = qContext.IPdict_microwave

    for key,value in IPdict.items():
        server.select_device(value)
        server.output(False)
    return
    

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



    
    
def RunAllExperiment(function,iterable,dataset,
                     collect: bool = True,
                     raw: bool = False,
                     pipesize: int = 10):
    """ Define an abstract loop to iterate a funtion for iterable

    Example:
    prepare parameters: scanning axes, dependences and other parameters
    start experiment with special sequences according to the parameters
    
    Args:
        function: give special sequences, parameters
        iterable: iterated over to produce values that are fed to function as parameters.
        dataset: zilabrad.pyle.datasaver.Dataset, object for data saving
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
    
    # create qubitContext (singleton) 
    qContext = qubitContext()
    
    results = dataset.capture(wrapped())
        
    resultArray = np.asarray(list(results))
    # for result in results:
    #     result_list.append(result)
    
    qContext.clearTempParas()
    
    if collect:
        return resultArray



def set_microwaveSource(freqList,powerList):
    """set frequency and power for microwaveSource devices
    """
    # print(freqList[0],powerList[0])

    qContext = qubitContext()
    server = qContext.servers_microwave
    IPdict = qContext.IPdict_microwave

    for i,key in enumerate(IPdict):
        server.select_device(IPdict[key])
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
    qContext = qubitContext()
    qa = qContext.servers_qa['qa_1']

    FS = qa.FS
    waveServer = waveforms.waveServer(device_id='0')
    
    wave_readout_func = [waveforms.NOTHING,waveforms.NOTHING]
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
    
    waveServer.set_tlist(start=start,end=end,fs=FS)
    wave_readout = [waveServer.func2array(wave_readout_func[i],start,end) for i in [0,1]]
    return wave_readout


def makeSequence_AWG(qubits):
    """
    waveServer: zilabrad.instrument.waveforms
    FS: sampling rates
    """
    qContext = qubitContext()
    hds = qContext.servers_hd
    hd = hds['hd_1']
    
    FS = hd.FS

    
    wave_AWG = []
    waveServer = waveforms.waveServer(device_id='0')
    
    ## This version consider all of the ports of AWG require the same sampling points


    ### ----bias_start previously -----  XY,Z, pulse quantum getes... ------ readout ----- bias_end ---
    q_ref = qubits[0]
    start = -Unit2SI(q_ref['bias_start'])
    end = Unit2SI(q_ref['bias_end']) + Unit2SI(q_ref['awgs_pulse_len']) + Unit2SI(q_ref['readout_len'])
    
    ## this is just set parameters, not really generate a list, which is slow
    waveServer.set_tlist(start,end,fs=FS)
    
    for q in qubits: 
        # the order must be 'dc,xy,z' ! match the order in QubitContext
        
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



def setup_wiring(__wiring_mode__=None):
    '''
    QA mode: 
        2 --> Rotation; 
        7 --> Integration; 
    HD mode: 
        0 --> 4*2 awgs; 
        1 --> 2*4 awgs; 
        2 --> 1*8 awgs; 
    '''
    qContext = qubitContext()
    if __wiring_mode__ == None:
        __wiring_mode__ = qContext.wiring
    
    for name,mode in __wiring_mode__.items():
        if 'qa' in name:
            qContext.servers_qa[name].set_qaSource_mode(mode)
        elif 'hd' in name:
            qContext.servers_hd[name].awg_grouping(mode)
        else:
            print('Unknown name (%r) with mode(%r)'%(name,mode))
    
def setupDevices(qubits):
    q_ref = qubits[0]
    # only run once in the whole experimental loop
    qContext = qubitContext()
    qa = qContext.servers_qa['qa_1']
    
    
    if 'isNewExpStart' not in q_ref:
        print('isNewExpStart, setupDevices')
        qContext.refresh()
        setup_wiring()
        ## int: sample number for one sweep point
        qa.set_result_samples(q_ref['stats'])
        
        ## only related with wiring and devices, delay between QA signal output and demodulation    
        qa.set_readout_delay(q_ref['readout_delay'])
        
        ## set qa pulse length in AWGs, and set same length for demodulate.
        qa.set_pulse_length(q_ref['readout_len'])

        ## delay between zurich HD and QA
        qa.set_adc_trig_delay(q_ref['bias_start']['s']+q_ref['experiment_length']) 
        
        
        ## set demodulate frequency for qubits if you need readout the qubit
        f_read = []
        for qb in qubits:
            if qb.get('do_readout'): ##in _q.keys():
                f_read += [qb.demod_freq]
        if len(f_read) == 0:
            raise Exception('Must set one readout frequency at least')
        qa.set_qubit_frequency(f_read) ##
        

        q_ref['isNewExpStart'] = False # actually it can be arbitrary value
        
    else:
        ## delay between zurich HD and QA
        ## for example: in T1 measurement
        qa.set_adc_trig_delay(q_ref['bias_start']['s']+q_ref['experiment_length'])
    return


def runDevices(qubits,wave_AWG,wave_readout):
    qContext = qubitContext()
    qas = qContext.servers_qa
    qa = qas['qa_1']
    hds = qContext.servers_hd

    # hd = hd['1']
    
    # t0=time.time()
    qubits_port = qContext.getPorts(qubits)
    wave_dict = awgWave_dict(ports=qubits_port,waves=wave_AWG)
    # print('wave_dict use %.3f s'%(time.time()-t0))

    ## send data packet to multiply devices
    # t0=time.time()
    qa.send_waveform(waveform=wave_readout)
    # print('qa.send_waveform use %.3f s'%(time.time()-t0))

    # t0=time.time()
    for dev_id,waveforms in wave_dict.items():
        for awg in range(4): ## default 4 awgs in every zi hdawgs
            port = list(waveforms[awg].keys())
            wave = list(waveforms[awg].values())
            # print(port)
            if len(wave) == 1 or len(wave) == 2:
                hds[dev_id].send_waveform(waveform=wave,awg_index=awg,port=port)
                # print('awg_index: %r, port: %r;\n hd waveform_length: %r'%(awg,port,hd.waveform_length))
                # hd.send_waveform(waveform=wave,awg_index=awg,port=port) ## one device case, use same hd

            elif len(wave) > 2:
                print('Too many port: %r'%port)
            # else:
            #     pass ## empty case should not send
    # print('hds.send_waveform use %.3f s'%(time.time()-t0))

    ## start to run experiment
    for name,hd in hds.items():
        if name not in wave_dict.keys():
            continue
        for k in range(4):
            if len(wave_dict[name][k])!=0:
                hd.awg_open(awgs_index=[k])
    qa.awg_open()## download experimental data
    # t0=time.time()
    _data = qa.get_data()
    # print('get_data use %.3f s'%(time.time()-t0))
    
    if qa.source == 7: ## single channels
        return _data
    else: ## double channels
        ks = range(int(len(_data)/2))
        get_doubleChannel = lambda k: _data[2*k]+1j*_data[2*k+1]
        data_doubleChannel = list(map(get_doubleChannel,ks))
        return data_doubleChannel
            
    



def awgWave_dict(ports,waves):
    """ Rewrite waveform sequence and port as dictionary, 
        follow device name to sort. Hold empty dict if no
        wave in device's port. 

        Return:
            port_dict = {'dev.id':[{port1:wave1,port2:wave2},{..},{..},{..}]}
    """
    port_dict = {}
    for k,wave in enumerate(waves):
        dev_name = ports[k][0]
        awg_index = (ports[k][1]+1) // 2 -1 ## awg_index: (1~8)-->(0,1,2,3)
        p_idx =  (ports[k][1]+1) % 2 +1   ## port: (1~8) --> (1,2)

        if dev_name not in port_dict.keys():
            port_dict[dev_name] = [{},{},{},{}]
        port_dict[dev_name][awg_index][p_idx] = wave

    return port_dict


def runQubits(qubits,exp_devices = None):
    """ generally for running multiqubits

    Args:
        qubits (list): a list of dictionary
        _runQ_servers (list/tuple): instances used to control device

    TODO:
    - clear q.xy/z/dc and their array after send();
    """
    # prepare wave packets
    # time0 = time.time()
    wave_AWG = makeSequence_AWG(qubits)
    wave_readout = makeSequence_readout(qubits)
    # print('wavePrepare use %.3f s'%(time.time()-time0))
    
    q_ref = qubits[0]
    # ## Now it is only two microwave sources, in the future, it should be modified
    set_microwaveSource(freqList = [q_ref['readout_mw_fc'],q_ref['xy_mw_fc']],
                        powerList = [q_ref['readout_mw_power'],q_ref['xy_mw_power']])
    
    
    
    setupDevices(qubits)
    # print('setupDevices use %.3f s'%(time.time()-time0))

    ## run AWG and reaout devices and get data
    # time0 = time.time()
    data = runDevices(qubits,wave_AWG,wave_readout)
    # print('get data use %.3f s'%(time.time()-time0))
    return data

