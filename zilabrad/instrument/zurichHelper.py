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

from zilabrad.util import singleton,singletonMany
from zilabrad.pyle.registry import RegistryWrapper
import labrad
from labrad.units import Unit,Value
_unitSpace = ('V', 'mV', 'us', 'ns','s', 'GHz', 'MHz','kHz','Hz', 'dBm', 'rad','None')
V, mV, us, ns,s, GHz, MHz,kHz,Hz, dBm, rad,_l  = [Unit(s) for s in _unitSpace]
import pyvisa

from zilabrad.instrument.waveforms import convertUnits,waveServer

logging.basicConfig(format='%(asctime)s | %(name)s [%(levelname)s] : %(message)s',
                    level=logging.INFO
                    )

'''
    TODO: add 'daq = zhinst.ziPython.ziDAQServer(ip,8004,6)' to create 
          py API object insteal of 'create_api_session'.
          Testing control API via IP/TCP from other PC.
            -- 2020.10.13 finished by hwh

'''






@singleton
class ziDAQ(object):
    def __init__(self,connectivity=8004,labone_ip='localhost'):
        ## connectivity must 8004 for zurish instruments
        self.daq = zhinst.ziPython.ziDAQServer(labone_ip,connectivity,6)
    def secret_mode(self,mode=1):
        ## mode = 1: daq can be created by everyone
        ## mode = 0: daq only be used by localhost
        self.daq.setInt('/zi/config/open',mode) 

@singletonMany
class zurich_qa(object):
    def __init__(self,obj_name='QA_1',device_id='dev2592',labone_ip='localhost'): 
        self.obj_name = obj_name
        self.id = device_id
        self.noisy = False ## 开启的话, 会打开所有正常工作时的print语句
        try:
            print('Bring up %s in %s'%(self.id,labone_ip))
            self.daq = ziDAQ().daq # connectivity = 8004
            self.daq.connectDevice(self.id,'1gbe')
            print(self.daq)
            self.FS = 1.8e9/(self.daq.getInt('/{:s}/awgs/0/time'.format(self.id))+1) ## 采样率
            self.init_setup()
        except:
            print('初始化失败，请检查仪器')

    def init_setup(self):
        """ initialize device settings.  
        """
        self.average = 1 # default 1, not average in device
        self.result_samples = 128
        self.qubit_frequency = [] # all demodulate frequency value
        self.paths = [] # save result path, equal to channel number
        self.source = 7 ## qa result mode, 7=integration--> return origin (I+iQ)
        self.waveform_length = 0 ## qa pulse length in AWGs; unit: sample number
        self.integration_length = 0 ## qa integration length; unit: sample number

        self.set_adc_trig_delay(0) ## delay after hd trigger, unit -> second
        self.set_readout_delay(0)  ## delay after qa pulse start, unit -> second
        self.set_pulse_length(0)   ## length of qa's awgs and demodulation, unit --> second

        # set deskew matrix  
        self.daq.setDouble('/{:s}/qas/0/deskew/rows/0/cols/0'.format(self.id), 1)
        self.daq.setDouble('/{:s}/qas/0/deskew/rows/1/cols/1'.format(self.id), 1)
        self.daq.setDouble('/{:s}/qas/0/deskew/rows/1/cols/0'.format(self.id), 1)
        self.daq.setDouble('/{:s}/qas/0/deskew/rows/0/cols/1'.format(self.id), 1)
        # set integration part
        self.daq.setInt('/{:s}/qas/0/integration/mode'.format(self.id), 0) ## set 0 -> standard mode to integration
        # set output part 
        self.daq.setInt('/{:s}/awgs/0/outputs/*/mode'.format(self.id),0) ## awgs output mode: 0=plain
        self.daq.setDouble('/{:s}/sigouts/*/range'.format(self.id), 1.5) # output range 1.5 V_peak
        # set input part
        self.daq.setDouble('/{:s}/sigins/*/range'.format(self.id), 1) # input range 1V 
        self.daq.setInt('/{:s}/sigins/*/imp50'.format(self.id), 1) # 50 Ohm input impedance
        ## set trigger part
        self.daq.setInt('/{:s}/triggers/out/0/source'.format(self.id), 32) ## open trig 1
        self.daq.setInt('/{:s}/triggers/out/1/source'.format(self.id), 33) ## open trig 2
        self.daq.setInt('/{:s}/triggers/out/*/drive'.format(self.id), 1)
        # set DIO output as qubit result
        self.daq.setInt('/{:s}/dios/0/drive'.format(self.id), 15)
        self.daq.setInt('/{:s}/dios/0/mode'.format(self.id), 2)
        self.daq.setInt('/{:s}/dios/0/extclk'.format(self.id), 2) # Sample DIO data at 50 MHz

        self.daq.setInt('/{:s}/sigouts/*/on'.format(self.id), 1) ## run = 1 open AWG, run =0 close AWG
        self.daq.setInt('/{:s}/awgs/0/single'.format(self.id), 1) ## close Rerun
        print('%s: Complete Initialization'%self.id)

    def reset_zi(self):
        print('关机重启吧！！')

    ####-- device parameters set & get --####
    @convertUnits(delay='s')
    def set_adc_trig_delay(self,delay):
        ''' delay: hd pulse start --> qa pulse start

            Here convert value from second to sample number, 
            8 samples as a unit.
        '''
        _adc_trig_delay_ = int(delay*self.FS/8)*8 ## unit -> Sample Number
        self.daq.setDouble('/{:s}/awgs/0/userregs/0'.format(self.id), _adc_trig_delay_/8) # send to device
            
    @convertUnits(readout_delay='s')
    def set_readout_delay(self,readout_delay):
        ''' delay: qa pulse start --> qa integration start

            Here convert value from second to sample number, 
            4 samples as a unit.
        '''
        delay_sample = int(readout_delay*self.FS/4)*4 ## unit: Sample Number
        # send to device
        self.daq.setDouble('/{:s}/qas/0/delay'.format(self.id),delay_sample) # send to device
    
    @convertUnits(length='s')
    def set_pulse_length(self, length):
        ''' length: set qa pulse length in AWGs, 
                    and set same length for demodulate.

            INPUT: unit --> Second, 
            SAVE: unit --> Sample number

            Demodulate has maximum length 4096.
            Ignore exceeding part. 
        '''
        if length > 4096/1.8:
            print('QA pulse lenght too long ( %.1f>%.1f ns)'%(length,4096/1.8))
            self.waveform_length = 4096 ## set the maximum length
        else:
            self.waveform_length =  int(length*self.FS/8)*8 ## unit --> Sample Number
        self.integration_length = self.waveform_length ## unit --> Sample Number

    def update_pulse_length(self):
        ''' Update self.waveform_length value via 
            daq.getList(), make sure send_waveform()
            having true length.
        '''
        qainfo = self.daq.getList('/{:s}/awgs/0/waveform/waves/0'.format(self.id))
        if len(qainfo)==0:
            self.waveform_length = 0
        elif len(qainfo)==1:
            self.waveform_length = int(len(qainfo[0][1][0]['vector'])/2) ## qalen has double channel wave;
        else:
            raise Exception('Unknown QA infomation:\n',qainfo)

    def awg_open(self):
        self.daq.syncSetInt('/{:s}/awgs/0/enable'.format(self.id), 1)
        if self.noisy:
            print('\n AWG running. \n')

    ####-- AWGs waveform --####
    def awg_builder(self,waveform=[[0],[0]],awg_index=0):
        """ Build waveforms sequencer. Then compile and send it to devices.
        """
        # create waveform
        define_str = 'wave wave0 = vect(_w_);\n'
        str0 = ''
        play_str = ''
        for i in range(len(waveform)):
            # wave wave1 = vect(_w1); wave wave2 = vect(_w2_);...
            str0 = str0 + define_str.replace('0', str(i+1)).replace('_w_', ','.join([str(x) for x in waveform[i]]))
            # 1, w1, 2, w2, ....
            play_str = play_str + (  ','+ str(i+1) + ',wave'+str(i+1) )
        
        play_str = play_str[1:] # delect the first comma
        
        awg_program = textwrap.dedent("""\
        $str0       
        const f_s = _c0_;

        setTrigger(AWG_INTEGRATION_ARM);// initialize integration

        setTrigger(0b000);
        repeat(_c3_){    // = qa.average
        repeat (_c2_) {  // = qa.result_samples
                // waitDigTrigger(1,1);
                setTrigger(0b11); // trigger output: rise
                wait(10e-9*f_s/8); // trigger length: 10 ns / 18 samples
                setTrigger(0b00); // trigger output: fall
                wait(getUserReg(0)); // wait time -> adc_trig_delay
                playWave($play_str);
                setTrigger(AWG_INTEGRATION_ARM + AWG_INTEGRATION_TRIGGER + AWG_MONITOR_TRIGGER);// start demodulate
                setTrigger(AWG_INTEGRATION_ARM);// reset intergration
                waitWave();
                wait(200e-6*f_s/8); // wait 200us
        }
        }        
        """)
        awg_program = awg_program.replace('_c0_', str(self.FS))
        awg_program = awg_program.replace('_c2_', str(self.result_samples))
        awg_program = awg_program.replace('_c3_', str(self.average))
        awg_program = awg_program.replace('$str0', str0)
        awg_program = awg_program.replace('$play_str', play_str)
        self.awg_upload_string(awg_program,awg_index=awg_index)
        self.update_pulse_length() ## updata self.waveform_lenght

    def awg_upload_string(self,awg_program,awg_index=0): 
        """ awg_program: waveforms sequencer text
            awg_index: this device's awgs sequencer index. 
                       If awgs grouping == 4*2, this index 
                       can be selected as 0,1,2,3

            write into waveforms sequencer and compile it.
        """
        awgModule = self.daq.awgModule() ## this API needs 0.2s to create
        awgModule.set('awgModule/device', self.id)
        awgModule.set('awgModule/index', awg_index)
        awgModule.execute() ## Starts the awgModule if not yet running.
        awgModule.set('awgModule/compiler/sourcestring', awg_program) ## to compile
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
            while awgModule.getDouble('awgModule/progress') < 1.0:
                time.sleep(0.1)
        if self.noisy:
            print('\n AWG upload successful. Output enabled. AWG Standby. \n')

    def send_waveform(self,waveform=[[0],[0]]):
        """ waveform: all waveform in this device

            Here judge which awgs or ports will be used
            to reload. Fill zeros at the end of waveform 
            to match the prior waveform length or compile 
            sequencer again.
        """
        _n_ = self.waveform_length - len(waveform[0])
        if _n_ >= 0:
            try:
                waveform_add = [np.hstack((wf,np.zeros(_n_))) for wf in waveform] 
                self.reload_waveform(waveform=waveform_add)
            except:
                self.update_pulse_length()
                _n_ = self.waveform_length - len(waveform[0])
                if _n_ >= 0:
                    waveform_add = [np.hstack((wf,np.zeros(_n_))) for wf in waveform] 
                    self.reload_waveform(waveform=waveform_add)
                else:
                    print('New QA waveform(len=%r > %r)'%(len(waveform[0]),self.waveform_length))
                    t0 = time.time()
                    self.awg_builder(waveform=waveform,awg_index=0)
                    print('QA-awg_builder:%.3f'%(time.time()-t0))
        else:
            print('New QA waveform(len=%r > %r)'%(len(waveform[0]),self.waveform_length))
            t0 = time.time()
            self.awg_builder(waveform=waveform,awg_index=0)
            print('QA-awg_builder:%.3f'%(time.time()-t0))

    def reload_waveform(self,waveform,awg_index=0,index=0): 
        """ waveform: (numpy.array) one/two waves with unit amplitude.
            awg_index: this devices awg sequencer index
            index: this sequencer output port index
        """
        waveform_native = convert_awg_waveform(waveform)
        path = '/{:s}/awgs/{:d}/waveform/waves/{:d}'.format(self.id,awg_index,index)
        self.daq.setVector(path, waveform_native)



    ####--set qa demodulation frequency (qubit sideband part)--####
    def set_qubit_frequency(self, frequency_list):
        # set integration weights, and result paths
        self.qubit_frequency = frequency_list
        self.set_all_integration() ## set integration weights
        self.set_subscribe() ## set result paths

    def set_all_integration(self):
        for i in range(len(self.qubit_frequency)):
            self.set_qubit_integration_I_Q(i,self.qubit_frequency[i])

    def set_qubit_integration_I_Q(self, channel, qubit_frequency): 
        # assign real and image integration coefficient 
        # integration settings for one I/Q pair
        from numpy import pi
        self.daq.setDouble('/{:s}/qas/0/integration/length'.format(self.id), self.integration_length)
        w_index      = np.arange(0, self.integration_length , 1)
        weights_real = np.cos(w_index/1.8e9*qubit_frequency*2*pi)
        weights_imag = np.sin(w_index/1.8e9*qubit_frequency*2*pi)
        w_real = np.array(weights_real)
        w_imag = np.array(weights_imag)
        self.daq.setVector('/{:s}/qas/0/integration/weights/{}/real'.format(self.id, channel), w_real)
        self.daq.setVector('/{:s}/qas/0/integration/weights/{}/imag'.format(self.id, channel), w_imag)
        # set signal input mapping for QA channel : 0 -> 1 real, 2 imag
        self.daq.setInt('/{:s}/qas/0/integration/sources/{:d}'.format(self.id, channel), 0)


    ####--readout result part--####
    def set_subscribe(self,source=None): 
        """ set demodulate result parameters --> upload qa result's paths
        """
        if source==None:
            source = self.source
        self.daq.setInt('/{:s}/qas/0/result/length'.format(self.id), self.result_samples)#results length 
        self.daq.setInt('/{:s}/qas/0/result/averages'.format(self.id), self.average)# average results
        
        self.daq.setInt('/{:s}/qas/0/result/reset'.format(self.id), 1) ## reset qa result 
        self.daq.setInt('/{:s}/qas/0/result/enable'.format(self.id), 1) ## stert qa result module, wait value
        # self.daq.sync() ## wait setting 
        self.daq.setInt('/{:s}/qas/0/result/source'.format(self.id), source) # integration=7

        self.paths = []
        for ch in range(len(self.qubit_frequency)):
            path = '/{:s}/qas/0/result/data/{:d}/wave'.format(self.id,ch)
            self.paths.append(path)

        if self.noisy:
            print('\n', 'Subscribed paths: \n ', self.paths, '\n')
        self.daq.subscribe(self.paths)

    def stop_subscribe(self):       
        # Stop result unit
        self.daq.unsubscribe(self.paths)
        self.daq.setInt('/{:s}/qas/0/result/enable'.format(self.id), 0)

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

    def get_data(self):
        data = self.acquisition_poll(self.daq, self.paths, self.result_samples, timeout=10)
        val,chan = [],0
        for path, samples in data.items():
            val.append(samples)
            chan += 1
        return val






@singletonMany
class zurich_hd:
    def __init__(self,obj_name = 'HD_1',device_id='dev8334',labone_ip='localhost'):   
        self.id = device_id
        self.obj_name = obj_name
        self.noisy = False ## 开启的话, 会打开所有正常工作时的print语句
        try:
            print('Bring up %s in %s'%(self.id,labone_ip))
            self.daq = ziDAQ(8004).daq # connectivity = 8004
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

    def reset_zi(self):
        print('关机重启吧！！')

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
        if len(hdinfo)==0:
            self.waveform_length = 0
        elif len(hdinfo)==1:
            self.waveform_length = int(len(hdinfo[0][1][0]['vector'])/2) ## qalen has double channel wave;
        else:
            raise Exception('Unknown QA infomation:\n',qainfo)
       
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




