# -*- coding: utf-8 -*-
"""
Helpers for Zurich Instruments

including classes for AWG devices, microwave sources
"""

import logging
import zhinst.utils ## create API object
import textwrap ## to write sequencer's code
import time ## show total time in experiments
import matplotlib.pyplot as plt ## give picture
import numpy as np 
from math import ceil,pi

from zilabrad.pyle.registry import RegistryWrapper
import labrad
from labrad.units import Unit,Value
_unitSpace = ('V', 'mV', 'us', 'ns','s', 'GHz', 'MHz','kHz','Hz', 'dBm', 'rad','None')
V, mV, us, ns,s, GHz, MHz,kHz,Hz, dBm, rad,_l  = [Unit(s) for s in _unitSpace]
import pyvisa

from zilabrad.instrument.waveforms import convertUnits,waveform

logging.basicConfig(format='%(asctime)s | %(name)s [%(levelname)s] : %(message)s',
                    level=logging.INFO
                    )

'''
    TODO: add 'daq = zhinst.ziPython.ziDAQServer(ip,8004,6)' to create 
          py API object insteal of 'create_api_session'.
          Testing control API via IP/TCP from other PC.
            -- 2020.10.13 finished by hwh
'''




def _call_object_func(server: object,_name_eval: dict,command: None or str = None)->bool:
    """  call an object accroding to a dict

    Example:
        _name_eval = {
        'zurich_qa':'init_setup'}
        
        _call_object_func(qa,_name_eval) # qa is instance of zurich_qa
        equals to: qa.init_setup()

    Returns:
        False if server name not in _name_eval.keys()
        else: True
    """
    name = server.__class__.__name__

    if command == None:
        command = r'server.' + _name_eval[name] + '()'
    
    if name not in _name_eval.keys():
        logging.info("some of args of server:object (qa,hd...) invalid")
        return False
    else:
        try:
            eval(command)
            logging.info(name+" successfully call %s",_name_eval[name])
        except:
            logging.warning(name+"cannot call %s",_name_eval[name])
        return True



def _init_device(*servers:object):
    """  initialize all server setup; 
    Example: _init_device(qa,hd,mw,mw_r)
    """
    _name_eval = {
    'zurich_qa':'init_setup',
    'zurich_hd':'init_setup',
    'microwave_source':'init_setup',
    }

    for server in servers:
        _call_object_func(server,_name_eval)
    return


def _check_device(*servers:object):
    """
    Make sure all device in work before runQ, or bringup again
    Args:
        *servers: one or more server, server is an instance
    Example: 
        _check_device(mw,mw_r)
        mw,mw_r are instances of microwave_source
    """
    _name_eval = {
    'microwave_source':'refresh',
    }
    for server in servers:
        _call_object_func(server,_name_eval)
    return


def _stop_device(*servers:object):
    """  close all device; 
    Args:
        *servers: one or more server, server is an instance
    Example: 
        _stop_device(qa,hd)
        qa,hd: instances of zurich_qa and zurich_hd
    """
    _name_eval = {
    'zurich_qa':'stop_subscribe',
    'zurich_hd':'awg_close_all',
    }

    for server in servers:
        _call_object_func(server,_name_eval)
    return


def _mpAwg_init(qubits:list,servers):
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
    qa,hd = servers[:2]
    q = qubits[0] ## the first qubit as master

    hd.pulse_length_s = 0 ## add hdawgs length with unit[s]

    qa.result_samples = qubits[0]['stats']  ## int: sample number for one sweep point
    qa.set_adc_trig_delay(q['bias_start']+hd.pulse_length_s*s, q['readout_delay'])
    qa.set_pulse_length(q['readout_len'])

    f_read = []
    for qb in qubits:
        if qb['do_readout']: ##in _q.keys():
            f_read += [qb.demod_freq]
    if len(f_read) == 0:
        raise Exception('Must set one readout frequency at least')
    qa.set_qubit_frequency(f_read)

    ## initialize waveforms and building 
    qa.update_wave_length()
    hd.update_wave_length()
    ### ----- finish ----- ###
    return 


class zurich_qa(object):
    def __init__(self,device_id,labone_ip='localhost'):   
        self.id = device_id
        self.noisy = False ## 开启的话, 会打开所有正常工作时的print语句
        try:
            print('Bring up %s in %s'%(self.id,labone_ip))
            self.daq = zhinst.ziPython.ziDAQServer(labone_ip,8004,6)
            self.daq.connectDevice(self.id,'1gbe')
            print(self.daq)
            self.FS = 1.8e9/(self.daq.getInt('/{:s}/awgs/0/time'.format(self.id))+1) ## 采样率
            self.init_setup()
        except:
            print('初始化失败，请检查仪器')

    def init_setup(self):
        self.average = 1 #不用硬件平均。保持默认不变
        self.result_samples = 100
        self.channels = [] # channel from 1 to 10;
        self.qubit_frequency = [] # match channel number
        self.paths = [] # save result path 
        self.source = 7 ## 解调模式, 7=integration; 
        self.waveform_length = 0 ## unit: sample number

        self.set_adc_trig_delay(0) ## delay after trigger; unit -> second
        self.set_pulse_length(0)   ## length of qa's awg, unit --> second

        self.daq.setInt('/{:s}/qas/0/integration/mode'.format(self.id), 0) ## set standard mode to integration
        self.daq.setDouble('/{:s}/qas/0/delay'.format(self.id), 0); # delay 0 samples for integration
        # 跳过crosstalk 操作，节约时间
        self.daq.setInt('/{:s}/qas/0/crosstalk/bypass'.format(self.id), 1)
        # set deskew matrix  
        self.daq.setDouble('/{:s}/qas/0/deskew/rows/0/cols/0'.format(self.id), 1)
        self.daq.setDouble('/{:s}/qas/0/deskew/rows/1/cols/1'.format(self.id), 1)
        self.daq.setDouble('/{:s}/qas/0/deskew/rows/1/cols/0'.format(self.id), 1)
        self.daq.setDouble('/{:s}/qas/0/deskew/rows/0/cols/1'.format(self.id), 1)
        #设置输出量程
        self.daq.setInt('/{:s}/awgs/0/outputs/*/mode'.format(self.id),0) ## output mode: plain
        self.daq.setDouble('/{:s}/sigouts/*/range'.format(self.id), 1) # output 1.0 V_peak
        #设置输入量
        self.daq.setDouble('/{:s}/sigins/*/range'.format(self.id), 1) # 输入量程1V
        self.daq.setInt('/{:s}/sigins/*/imp50'.format(self.id), 1) # 50 Ohm 输入阻抗
        
        self.daq.setInt('/{:s}/triggers/out/0/source'.format(self.id), 32) ## open trig 1
        self.daq.setInt('/{:s}/triggers/out/1/source'.format(self.id), 33) ## open trig 2
        self.daq.setInt('/{:s}/triggers/out/*/drive'.format(self.id), 1)
        # 设置DIO输出为qubit result
        self.daq.setInt('/{:s}/dios/0/drive'.format(self.id), 15)
        self.daq.setInt('/{:s}/dios/0/mode'.format(self.id), 2)
        # Sample DIO data at 50 MHz
        self.daq.setInt('/{:s}/dios/0/extclk'.format(self.id), 2)

        # self.daq.setInt('/{:s}/qas/0/result/length'.format(self.id), self.result_samples) #results length 
        # self.daq.setInt('/{:s}/qas/0/result/averages'.format(self.id), self.average) # average results
        #以下会设置读取脉冲的参数
        # self.daq.setDouble('/{:s}/awgs/0/userregs/0'.format(self.id), self.average)# average results
        # self.daq.setDouble('/{:s}/awgs/0/userregs/1'.format(self.id), self.result_samples)# #results length
        """打开AWG， 打开输出，设置输出量程"""
        # run = 1 start AWG, run =0 close AWG
        self.daq.setInt('/{:s}/sigouts/*/on'.format(self.id), 1)
        self.daq.setInt('/{:s}/awgs/0/single'.format(self.id), 1)
        # Arm the device
        self.daq.sync() ## 同步设置
        print('%s: Complete Initialization'%self.id)


    ####--AWG波形设置部分--####
    @convertUnits(delay='s', readout_delay='s')
    def set_adc_trig_delay(self,delay,readout_delay=None):
        self.adc_trig_delay_s = int(delay*self.FS/8)*8/self.FS ## unit: second
        self._adc_trig_delay_ = self.adc_trig_delay_s*self.FS ## delay from dc trigger to ADC trigger; unit -> Sample Number
        self.daq.setDouble('/{:s}/awgs/0/userregs/0'.format(self.id), self._adc_trig_delay_/8) # qa._adc_trig_delay_
        if readout_delay is not None:
            self.set_readout_delay(readout_delay)
            
    @convertUnits(readout_delay_s='s')
    def set_readout_delay(self,readout_delay_s):
        delay_sample = int(readout_delay_s*self.FS/4)*4 ## unit: Sample Number
        self.daq.setDouble('/{:s}/qas/0/delay'.format(self.id),delay_sample)
    
    @convertUnits(length='s')
    def set_pulse_length(self, length):
        if length > 4096/1.8:
            print('QA pulse lenght too long ( %.1f>%.1f ns)'%(length,4096/1.8))
            length = 4096/1.8 ## set the maximum length
        self.pulse_length_s = int(length*self.FS/8)*8/self.FS
        self._pulse_length_ =  self.pulse_length_s*self.FS ## unit --> Sample Number
        self._integration_length_ = self._pulse_length_ ## unit --> Sample Number

    def awg_builder(self, waveform = [[0],[0]], trigger = 0):
        """根据构建一个awg 程序， waveform的波形同时播放；可以工作在被触发模式 trigger =1 """       
        # #检查波形每个点的值是否超过1、波形是没有单位的，不应该超过1
        # assert np.max(waveform) <= 1, '\n waveform value is dimensionless, less than 1. Check waveform values before proceed.\n'     
        #构建波形
        define_str = 'wave wave0 = vect(_w_);\n'
        str0 = ''
        play_str = ''
        for i in range(len(waveform)):
            # wave wave1 = vect(_w1); wave wave2 = vect(_w2_);...
            str0 = str0 + define_str.replace('0', str(i+1)).replace('_w_', ','.join([str(x) for x in waveform[i]]))
            # 1, w1, 2, w2, ....
            play_str = play_str + (  ','+ str(i+1) + ',wave'+str(i+1) )
        
        play_str = play_str[1:]
        
        awg_program = textwrap.dedent("""\
        $str0       
        const f_s = _c0_;

        setTrigger(AWG_INTEGRATION_ARM);// 启动解调器

        setTrigger(0b000);
        repeat(_c3_){    // = qa.average
        repeat (_c2_) {  // = qa.result_samples
                // waitDigTrigger(1,1);
                setTrigger(0b11); // trigger other device: rise
                wait(10e-9*f_s/8); // trigger长度为10ns
                setTrigger(0b00); // trigger other device: fall
                wait(getUserReg(0)); // wait time = qa._adc_trig_delay_
                playWave($play_str);
                setTrigger(AWG_INTEGRATION_ARM + AWG_INTEGRATION_TRIGGER + AWG_MONITOR_TRIGGER);// 开始抓取信号，进行integration
                setTrigger(AWG_INTEGRATION_ARM);//重置
                waitWave();
                wait(200e-6*f_s/8); //等待200us后发送下一个
        }
        //wait(200e-6*f_s/8); // 间隔200us
        }        
        """)
        awg_program = awg_program.replace('_c0_', str(self.FS))
        # awg_program = awg_program.replace('_c1_', 'getUserReg(0)')
        awg_program = awg_program.replace('_c2_', str(self.result_samples))
        awg_program = awg_program.replace('_c3_', str(self.average))
        awg_program = awg_program.replace('$str0', str0)
        awg_program = awg_program.replace('$play_str', play_str)
        self.awg_upload_string(awg_program)
        ## updata waveform lenght infomation
        self.update_wave_length()

    def awg_upload_string(self,awg_program, awg_index = 0): 
        """"写入波形并编译
        #awg_prgram 是一个字符串，AWG 程序
        #awg_index 是AWG的序列号， 当grouping 是4x2的时候，有4个AWG. awg_index = 0 时，即第一个AWG, 控制第一和第二个通道"""
        awgModule = self.daq.awgModule()
        awgModule.set('awgModule/device', self.device)
        awgModule.set('awgModule/index', awg_index)# AWG 0, 1, 2, 3
        awgModule.execute()
        awgModule.set('awgModule/compiler/sourcestring', awg_program)
        while awgModule.getInt('awgModule/compiler/status') == -1:
            time.sleep(0.1)
        # Ensure that compilation was successful
        if awgModule.getInt('awgModule/compiler/status') == 1:
            # compilation failed, raise an exception
            raise Exception(awgModule.getString('awgModule/compiler/statusstring'))
        else:
            if awgModule.getInt('awgModule/compiler/status') == 0 and self.noisy:
                print("Compilation successful with no warnings, will upload the program to the instrument.")
            if awgModule.getInt('awgModule/compiler/status') == 2 and self.noisy:
                print("Compilation successful with warnings, will upload the program to the instrument.")
                print("Compiler warning: ", awgModule.getString('awgModule/compiler/statusstring'))
            # wait for waveform upload to finish
            i = 0
            while awgModule.getDouble('awgModule/progress') < 1.0:
                time.sleep(0.1)
                i += 1
        if self.noisy:
            print('\n AWG upload successful. Output enabled. AWG Standby. \n')


    def reload_waveform(self,waveform,awg_index=0,index=0): ## (dev_id, awg编号(看UI的awg core), index是wave编号;)
        waveform_native = convert_awg_waveform(waveform)
        path = '/{:s}/awgs/{:d}/waveform/waves/{:d}'.format(self.id,awg_index,index)
        self.daq.setVector(path, waveform_native)


    def awg_open(self):
        self.daq.syncSetInt('/{:s}/awgs/0/enable'.format(self.id), 1)
        if self.noisy:
            print('\n AWG running. \n')

    def update_wave_length(self):
        qainfo = self.daq.getList('/{:s}/awgs/0/waveform/waves/0'.format(self.id))
        self.waveform_length = int(len(qainfo[0][1][0]['vector'])/2) ## qalen has double channel wave;

    def send_waveform(self,waveform=[[0],[0]],check=False):
        if check:
            update_wave_length()
        _n_ = self.waveform_length - len(waveform[0])
        if _n_ >= 0:
            waveform_add = [np.hstack((wf,np.zeros(_n_))) for wf in waveform] 
            self.reload_waveform(waveform=waveform_add)
        else:
            print('New QA waveform(len=%r > %r)'%(len(waveform[0]),self.waveform_length))
            t0 = time.time()
            self.awg_builder(waveform=waveform)
            print('QA-awg_builder:%.3f'%(time.time()-t0))



    ####--设置比特的读取频率sideband部分--####
    def set_qubit_frequency(self, frequency_array):
        #设置解调用的weights, 更新 channels, 和paths
        self.channels = [] # channel from 0 to 9;
        # if frequency_array != self.qubit_frequency: ## 没有改动就不重新上传
        self.qubit_frequency = frequency_array
        self.channels = [k for k in range(len(self.qubit_frequency))]  #更新解调器/通道, channel from 0 to 9;
        self.set_all_integration()
        self.set_subscribe()

    def set_all_integration(self):
        for i in range(len(self.qubit_frequency)):
            self.set_qubit_integration_I_Q(i,self.qubit_frequency[i])

    def set_qubit_integration_I_Q(self, channel, qubit_frequency): 
        # assign real and image integration coefficient 
        # integration settings for one I/Q pair
        from numpy import pi
        self.daq.setDouble('/{:s}/qas/0/integration/length'.format(self.id), self._integration_length_)
        w_index      = np.arange(0, self._integration_length_ , 1)
        weights_real = np.cos(w_index/1.8e9*qubit_frequency*2*pi)
        weights_imag = np.sin(w_index/1.8e9*qubit_frequency*2*pi)
        w_real = np.array(weights_real)
        w_imag = np.array(weights_imag)
        self.daq.setVector('/{:s}/qas/0/integration/weights/{}/real'.format(self.id, channel), w_real)
        self.daq.setVector('/{:s}/qas/0/integration/weights/{}/imag'.format(self.id, channel), w_imag)
        # set signal input mapping for QA channel : 0 -> 1 real, 2 imag
        self.daq.setInt('/{:s}/qas/0/integration/sources/{:d}'.format(self.id, channel), 0)


    ####--读取数据部分--####
    def acquisition_poll(self, daq, paths, num_samples, timeout=10.0):
        """ Polls the UHFQA for data.
    
        Args:
            paths (list): list of subscribed paths
            num_samples (int): expected number of samples
            timeout (float): time in seconds before timeout Error is raised.
        """
        if self.noisy:
            print('acquisition_poll')
        poll_length = 0.01  # s
        poll_timeout = 100  # ms
        poll_flags = 0
        poll_return_flat_dict = True
   
         # Keep list of recorded chunks of data for each subscribed path
        chunks = {p: [] for p in paths}
        gotem = {p: False for p in paths}
    
        # Poll data
        time = 0
        while time < timeout and not all(gotem.values()):
            if self.noisy:
                print('collecting results')
            dataset = daq.poll(poll_length, poll_timeout, poll_flags, poll_return_flat_dict)
            for p in paths:
                if p not in dataset:
                    continue
                for v in dataset[p]:
                    chunks[p].append(v['vector'])
                    num_obtained = sum([len(x) for x in chunks[p]])
                    if num_obtained >= num_samples:
                        gotem[p] = True
            time += poll_length
    
        if not all(gotem.values()):
            for p in paths:
                num_obtained = sum([len(x) for x in chunks[p]])
                print('Path {}: Got {} of {} samples'.format(p, num_obtained, num_samples))
            raise Exception('Timeout Error: Did not get all results within {:.1f} s!'.format(timeout))
    
        # Return dict of flattened data
        return {p: np.concatenate(v) for p, v in chunks.items()}      

    def stop_subscribe(self):       
        # Stop result unit
        self.daq.unsubscribe(self.paths)
        self.daq.setInt('/{:s}/qas/0/result/enable'.format(self.id), 0)
        self.qubit_frequency = []

    def set_subscribe(self,source=None): 
        """配置采集数据所需设置,设定读取result的path
        """
        if source==None:
            source = self.source
        self.paths = []
        self.daq.setInt('/{:s}/qas/0/result/length'.format(self.id), self.result_samples)#results length 
        self.daq.setInt('/{:s}/qas/0/result/averages'.format(self.id), self.average)# average results
        
        self.daq.setInt('/{:s}/qas/0/result/reset'.format(self.id), 1) ## reset
        self.daq.setInt('/{:s}/qas/0/result/enable'.format(self.id), 1) ##启动qa_result模块,等待数据
        self.daq.sync() ## 同步设置
        self.daq.setInt('/{:s}/qas/0/result/source'.format(self.id), source) # integration=7
        if self.noisy:
            print(self.channels)
        for ch in self.channels:
            path = '/{:s}/qas/0/result/data/{:d}/wave'.format(self.id,ch)
            self.paths.append(path)
            if self.noisy:
                print(path)
        if self.noisy:
            print('\n', 'Subscribed paths: \n ', self.paths, '\n')
        self.daq.subscribe(self.paths)
        if self.noisy:
            print('\n Acquiring data...\n')

    def get_data(self,do_plot=False,back=True):
        data = self.acquisition_poll(self.daq, self.paths, self.result_samples, timeout=10)
        
        val,chan = [],0
        for path, samples in data.items():
            val.append(samples)
            chan += 1

        if do_plot and self.source==7:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_title('Result')
            ax.set_xlabel('I (a.u.)')
            ax.set_ylabel('Q (a.u.)')
            for n1 in range(chan):
                plt.plot(np.real(val[n1]),np.imag(val[n1]),'.',label='Chan:%d'%n1)
            plt.legend(loc='best')
            fig.set_tight_layout(True)
        elif do_plot:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.set_title('Result unit')
            ax.set_ylabel('Amplitude (a.u.)')
            ax.set_xlabel('Measurement (#)')
            for path,samples in data.items():
                ax.plot(samples, label='{}'.format(path))
            plt.legend(loc='best')
            fig.set_tight_layout(True)

        if back:
            return val




class zurich_hd(object):
    def __init__(self,device_id,labone_ip='localhost'):   
        self.id = device_id
        self.noisy = False ## 开启的话, 会打开所有正常工作时的print语句
        try:
            print('Bring up %s in %s'%(self.id,labone_ip))
            self.daq = zhinst.ziPython.ziDAQServer(labone_ip,8004,6)
            self.daq.connectDevice(self.id,'1gbe')
            print(self.daq)
            self.pulse_length_s = 0 ## waveform length; unit --> Sample Number
            self.FS = 1.8e9/(self.daq.getInt('/{:s}/awgs/0/time'.format(self.id))+1) ## 采样率
            self.init_setup()
        except:
            print('初始化失败，请检查仪器')


    def init_setup(self):
        self.pulse_length_s = 0
        self.waveform_length = 0
        amplitude = 1.0
        exp_setting = [
            ['/%s/sigouts/*/on'               % (self.id), 1],
            ['/%s/sigouts/*/range'            % (self.id), 1],
            ['/%s/awgs/0/outputs/*/amplitude' % (self.id), amplitude],
            ['/%s/awgs/0/outputs/0/modulation/mode' % (self.id), 0],
            ['/%s/system/awg/channelgrouping'% self.id, 2],  ## use 1*8 channels
            ['/%s/awgs/0/time'                 % self.id, 0],
            ['/%s/awgs/0/userregs/0'           % self.id, 0],
            ['/%s/system/clocks/sampleclock/freq' % self.id, self.FS],
            ['/%s/system/clocks/referenceclock/source' % self.id, 1], ## set ref clock mode in 'External'
            ['/%s/awgs/0/dio/strobe/slope' % self.id, 0],
            ['/%s/awgs/0/dio/strobe/index' % self.id, 15],
            ['/%s/awgs/0/dio/valid/index' % self.id, 0],
            ['/%s/awgs/0/dio/valid/polarity' % self.id, 2],
            ['/%s/awgs/0/dio/mask/value' % self.id, 7], # 111 三位qubit results
            ['/%s/awgs/0/dio/mask/shift' % self.id, 1],
            ['/%s/raw/dios/0/extclk' % self.id, 2]
        ]
        self.daq.set(exp_setting)
        self.daq.setInt('/{:s}/awgs/0/auxtriggers/0/slope'.format(self.id), 1)
        self.daq.setInt('/{:s}/awgs/0/auxtriggers/0/channel'.format(self.id), 0)
        self.daq.setInt('/{:s}/triggers/in/0/imp50'.format(self.id), 0)  ## ???
        self.daq.setDouble('/{:s}/triggers/in/0/level'.format(self.id), 0.7) ### ???
        self.daq.setInt('/{:s}/awgs/0/single'.format(self.id), 1)
        # Ensure that all settings have taken effect on the device before continuing.
        self.daq.sync()
        print('%s: Complete Initialization'%self.id.upper())

    def awg_builder(self,waveform=[[0]],ports=[],awg_index=0):
        """
        根据波形构建一个awg 程序， waveform的波形同时播放，可以工作在被触发模式trigger =1 
        """
        #构建波形
        wave_len = len(waveform[0])
        define_str = 'wave wave0 = vect(_w_);\n'
        str0 = ''
        play_str = ''
        if len(ports) == 0:
            ports = np.arange(len(waveform))+1

        j = 0
        for i in ports:
            wf = waveform[j]
            # if len(wf)==2: ## 注意非bias的waveform就不要发送那么短的list了!!
            #     str0 = str0 + 'wave bias%i = %f*ones(%i);\n'%(i,float(wf[0]),int(wf[1]))
            #     play_str = play_str + (','+str(i)+',bias'+str(i))
            # else:
            str0 = str0 + define_str.replace('0', str(i)).replace('_w_', ','.join([str(x) for x in wf]))
            play_str = play_str + (  ','+ str(i) + ',wave'+str(i) )
            j += 1
        play_str = play_str[1:]
        
        awg_program = textwrap.dedent("""\
        const f_s = _c0_;
        $str0
        while(1){
        setTrigger(0b00);
        setTrigger(0b01);
        waitDigTrigger(1);
        playWave($play_str);
        waitWave();
        }
        """)
        awg_program = awg_program.replace('$str0', str0)
        awg_program = awg_program.replace('$play_str', play_str)
        awg_program = awg_program.replace('_c0_', str(self.FS))
        self.awg_upload_string(awg_program, awg_index)
        self.update_wave_length()

    def awg_upload_string(self,awg_program, awg_index = 0): 
        """"写入波形并编译
        #awg_prgram 是一个字符串，AWG 程序
        #awg_index 是AWG的序列号， 当grouping 是4x2的时候，有4个AWG. awg_index = 0 时，即第一个AWG, 控制第一和第二个通道"""
        awgModule = self.daq.awgModule()
        awgModule.set('awgModule/device', self.id)
        awgModule.set('awgModule/index', awg_index)# AWG 0, 1, 2, 3
        awgModule.execute()
        awgModule.set('awgModule/compiler/sourcestring', awg_program)
        while awgModule.getInt('awgModule/compiler/status') == -1:
            time.sleep(0.1)
        # Ensure that compilation was successful
        if awgModule.getInt('awgModule/compiler/status') == 1:
            # compilation failed, raise an exception
            raise Exception(awgModule.getString('awgModule/compiler/statusstring'))
        else:
            if awgModule.getInt('awgModule/compiler/status') == 0 and self.noisy:
                print("Compilation successful with no warnings, will upload the program to the instrument.")
            if awgModule.getInt('awgModule/compiler/status') == 2 and self.noisy:
                print("Compilation successful with warnings, will upload the program to the instrument.")
                print("Compiler warning: ", awgModule.getString('awgModule/compiler/statusstring'))
            # wait for waveform upload to finish
            i = 0
            while awgModule.getDouble('awgModule/progress') < 1.0:
                time.sleep(0.1)
                i += 1
        if self.noisy:
            print('\n AWG upload successful. Output enabled. AWG Standby. \n')

    def reload_waveform(self,waveform,awg_index=0,index=0): ## (dev_id, awg编号(看UI的awg core), index是wave的编号;)
        waveform_native = convert_awg_waveform(waveform)
        print(len(waveform_native))
        path = '/{:s}/awgs/{:d}/waveform/waves/{:d}'.format(self.id,awg_index,index)
        self.daq.setVector(path, waveform_native)

    def update_wave_length(self):
        hdinfo = self.daq.getList('/{:s}/awgs/0/waveform/waves/0'.format(self.id))
        self.waveform_length = int(len(hdinfo[0][1][0]['vector'])/2) ## qalen has double channel wave;
       
    def send_waveform(self,waveform=[[0],[0]],ports=[],check=False):
        if check:
            update_wave_length()      
        wave_dict = dict(zip(ports,waveform)) ## TODO: Aviod copying this wfs list;
        for k in range(8): 
            if k+1 not in ports:
                wave_dict[k+1]=np.zeros(len(waveform[0]))

        _n_ = self.waveform_length - len(waveform[0])
        if _n_ >= 0: ## if wave length enough to reload
            waveform_add = [np.hstack((wave_dict[k+1],np.zeros(_n_))) for k in range(8)] ## fill [0] to hold wf len;
            for k in range(4): ## reload wave following ports
                if k*2+1 in ports or k*2+2 in ports:
                    wf = [waveform_add[k*2],waveform_add[k*2+1]]
                    waveform_native = convert_awg_waveform(wf)
                    path = '/{:s}/awgs/{:d}/waveform/waves/{:d}'.format(self.id,k,0)
                    self.daq.setVector(path, waveform_native)
        else: ## if wave length too short, need building again;
            print('New HD waveform(len=%r > %r)'%(len(waveform[0]),self.waveform_length))
            waveform_send = [ _v_ for _v_ in wave_dict.values()]
            ports_send = [ _p_ for _p_ in wave_dict.keys()]
            t0 = time.time()
            self.awg_builder(waveform=waveform_send,ports=ports_send)
            print('HD-awg_builder:%.3f'%(time.time()-t0))


    def awg_open(self):
        """打开AWG， 打开输出，设置输出量程"""
        self.daq.setInt('/{:s}/awgs/{}/enable'.format(self.id, 0), 1)
        if self.noisy:
            print('\n AWG running. \n')

    def awg_close_all(self):
        self.daq.setInt('/{:s}/awgs/*/enable'.format(self.id), 0)


    def awg_grouping(self, grouping_index = 2):
        """配置AWG grouping 模式，确定有几个独立的序列编辑器。有多种组合"""
        #% 'system/awg/channelgrouping' : Configure how many independent sequencers
        #%   should run on the AWG and how the outputs are grouped by sequencer.
        #%   0 : 4x2 with HDAWG8; 2x2 with HDAWG4.
        #%   1: 2x4 with HDAWG8; 1x4 with HDAWG4.
        #%   2 : 1x8 with HDAWG8.
        grouping = ['4x2 with HDAWG8', '2x4 with HDAWG8', '1x8 with HDAWG8']
        self.daq.setInt('/{:s}/system/awg/channelgrouping'.format(self.id), grouping_index)
        self.daq.sync()
        if self.noisy:
            print('\n','HDAWG channel grouping:', grouping[grouping_index], '\n')
        



class microwave_source(object):
    def __init__(self,device_ip,object_name):
        self.ip = device_ip
        self.freq = 0
        self.power = 0
        self.rm = pyvisa.ResourceManager()
        self.bringup()
        self.object_name = object_name # mw, mw_r

    def init_setup(self):
        while self.get_output() == 0:
            self.set_output(1)


    @convertUnits(freq='Hz')
    def set_freq(self,freq):
        if self.freq != freq:
            self.dev.write(':sour:freq %f Hz'%freq)
            self.freq = freq

    @convertUnits(power='dBm')
    def set_power(self,power):
        if self.power != power:
            self.dev.write(':sour:pow %f'%power) 
            self.power = power

    def get_power(self):
        self.power = float(self.dev.query(':sour:pow?'))
        return self.power

    def get_freq(self):
        self.freq = float(self.dev.query(':sour:freq?'))
        return self.freq

    def get_output(self):
        state = self.dev.query(':outp:stat?')
        return int(state)

    def set_output(self,state):
        self.dev.write(':outp:stat %d'%state)

    def bringup(self):
        self.dev = self.rm.open_resource('TCPIP0::%s::inst0::INSTR' %self.ip)
        self.init_setup()
        logging.info('Bring up microwave_source:%s'%self.ip)

    def refresh(self):
        try:
            while self.get_output() == 0:
                self.set_output(1)
        except:
            self.bringup()
            logging.info(self.ip,'microwave_source bring up again.')


class yokogawa_dc_source(object):
    """docstring for yokogawa_dc_source"""
    def __init__(self,device_ip):
        self.ip = device_ip
        self.amp = 0.0 ## unit:V
        self.state = False
        rm = pyvisa.ResourceManager()
        self.dev = rm.open_resource('TCPIP0::%s::inst0::INSTR' %self.ip)
        self.init_setup()

    def init_setup(self):
        self.output(0)
        self.amp(self.amp)
    
    def sourFunc(self,mode=None):
        if mode is None:
            ans = self.dev.query('sour:func?')
        elif mode=='curr':
            dev.write('sens:func \'curr\'')
            dev.write('sour:func %s' % mode)
        elif mode=='volt':
            dev.write('sens:func \'volt\'')
            dev.write('sour:func %s' % mode)


    def amp(self,amp=None):
        ## unit:V
        if amp is None:
            self.amp = float(self.dev.query(':sour:lev?'))
            return self.amp
        else:
            self.amp = float(amp)
            self.dev.write(':sour:lev:auto %f'%self.amp) 


    def output(self,state=None):
        if state is None:
            self.state = int(self.dev.query(':outp:stat?'))
            return self.state
        else:
            self.state = state
            self.dev.write(':outp:stat %d'%self.state)





def convert_awg_waveform(wave_list):
    """
    Converts one or multiple arrays with waveform data to the native AWG
    waveform format (interleaved waves and markers as uint16).

    Waveform data can be provided as integer (no conversion) or floating point
    (range -1 to 1) arrays.

    Arguments:
      wave1 (array): Array with data of waveform 1.
      wave2 (array): Array with data of waveform 2.
      ...
    Returns:
      The converted uint16 waveform is returned.

    ** waveform reload needs each two channel one by one.
    """
    def uint16_waveform(wave): # Prepare waveforms
        wave = np.asarray(wave)
        if np.issubdtype(wave.dtype, np.floating):
            return np.asarray((np.power(2, 15) - 1) * wave, dtype=np.uint16)
        return np.asarray(wave, dtype=np.uint16)

    data_list = [uint16_waveform(w) for w in wave_list]
    waveform_data = np.vstack(tuple(data_list)).reshape((-2,), order='F')
    return waveform_data




