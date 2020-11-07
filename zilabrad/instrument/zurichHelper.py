# -*- coding: utf-8 -*-
"""
Helpers for Zurich Instruments
including classes for AWG devices, microwave sources
"""


import zhinst.utils ## create API object
import textwrap ## to write sequencer's code
import time ## show total time in experiments
import numpy as np 
from numpy import pi


from zilabrad.util import singleton,singletonMany
from zilabrad.instrument.waveforms import convertUnits





@singleton
class ziDAQ(object):
    """singleton class for zurich daq
    """
    def __init__(self,connectivity=8004,labone_ip='localhost'):
        ## connectivity must 8004 for zurish instruments
        self.daq = zhinst.ziPython.ziDAQServer(labone_ip,connectivity,6)
    def secret_mode(self,mode=1):
        ## mode = 1: daq can be created by everyone
        ## mode = 0: daq only be used by localhost
        self.daq.setInt('/zi/config/open',mode) 

@singletonMany
class zurich_qa(object):
    """server for zurich qa
    Args:
        obj_name (str): specify the object, the object with the same obj_name is singleton. 
        Example: If the class has been instantiated with obj_name=='obj1',  
        calling it again via 'zurich_qa(obj_name='obj1')' will not re-initiate but get the old object. 
    Instance: The instance now will return a dictionary of objects (value) and their obj_name (key) that has been created
    """
    def __init__(self,obj_name='QA_1',device_id='dev2592',labone_ip='localhost'): 
        self.obj_name = obj_name
        self.id = device_id
        self.noisy = False ## if open, will activate all print command during device working
        try:
            print('\nBring up %s in %s'%(self.id,labone_ip))
            self.daq = ziDAQ().daq
            self.daq.connectDevice(self.id,'1gbe')
            print(self.daq)
            self.FS = 1.8e9/(self.daq.getInt('/{:s}/awgs/0/time'.format(self.id))+1) ## sample rate
            self.init_setup()
        except:
            print('Failed to initialize, please check device.')

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
        print('You need to manually restart the device')

    ####-- device parameters set & get --####
    def set_result_samples(self,sample):
        """ sample: repetition number
            Meanwhile update repeat index in AWG sequencer
            and QA result parameter.
        """
        self.result_samples = int(sample)
        self.daq.setDouble('/{:s}/awgs/0/userregs/1'.format(self.id), self.result_samples) # send to device: Register 2
        self.daq.setInt('/{:s}/qas/0/result/length'.format(self.id), self.result_samples) # results length 

    @convertUnits(delay='s')
    def set_adc_trig_delay(self,delay):
        ''' delay: hd pulse start --> qa pulse start
            Here convert value from second to sample number, 
            8 samples as a unit.
        '''
        _adc_trig_delay_ = int(delay*self.FS/8)*8 ## unit -> Sample Number
        self.daq.setDouble('/{:s}/awgs/0/userregs/0'.format(self.id), _adc_trig_delay_/8) # send to device: Register 1
            
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
        if self.noisy:
            print('[%s] update_pulse_length: %r'%(self.id,self.waveform_length))

    def awg_open(self):
        self.daq.syncSetInt('/{:s}/awgs/0/enable'.format(self.id), 1)
        if self.noisy:
            print('\n AWG running. \n')

    def awg_close(self):       
        # Stop result unit
        self.daq.unsubscribe(self.paths)
        self.daq.setInt('/{:s}/qas/0/result/enable'.format(self.id), 0)

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
        repeat (getUserReg(1)) {  // = qa.result_samples
                // waitDigTrigger(1,1);
                setTrigger(0b11); // trigger output: rise
                wait(5); // trigger length: 22.2 ns / 40 samples
                setTrigger(0b00); // trigger output: fall
                wait(getUserReg(0)); // wait time -> adc_trig_delay
                playWave($play_str);
                setTrigger(AWG_INTEGRATION_ARM + AWG_INTEGRATION_TRIGGER + AWG_MONITOR_TRIGGER);// start demodulate
                setTrigger(AWG_INTEGRATION_ARM);// reset intergration
                waitWave();
                wait(200e-6*f_s/8); // wait 200us
        }       
        """)
        awg_program = awg_program.replace('_c0_', str(self.FS))
        # awg_program = awg_program.replace('_c2_', str(self.result_samples))
        # awg_program = awg_program.replace('_c3_', str(self.average))
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
            Here judge which awgs or port will be used
            to reload. Fill zeros at the end of waveform 
            to match the prior waveform length or compile 
            sequencer again.
        """
        _n_ = self.waveform_length - len(waveform[0])
        if _n_ >= 0:
            waveform_add = [np.hstack((wf,np.zeros(_n_))) for wf in waveform] 
            try:
                self.reload_waveform(waveform=waveform_add)
            except:
                self.update_pulse_length()
                _n_ = self.waveform_length - len(waveform[0])
                if _n_ >= 0:
                    waveform_add = [np.hstack((wf,np.zeros(_n_))) for wf in waveform] 
                    self.reload_waveform(waveform=waveform_add)
                else:
                    print('Bulid [%s-AWG0] Sequencer (len=%r > %r)'%(self.id,len(waveform[0]),self.waveform_length))
                    t0 = time.time()
                    self.awg_builder(waveform=waveform,awg_index=0)
                    print('[%s-AWG0] builder: %.3f s'%(self.id,time.time()-t0))
        else:
            print('Bulid [%s-AWG0] Sequencer (len=%r > %r)'%(self.id,len(waveform[0]),self.waveform_length))
            t0 = time.time()
            self.awg_builder(waveform=waveform,awg_index=0)
            print('[%s-AWG0] builder: %.3f s'%(self.id,time.time()-t0))
           

    def reload_waveform(self,waveform,awg_index=0,index=0): 
        """ waveform: (numpy.array) one/two waves with unit amplitude.
            awg_index: this devices awg sequencer index
            index: this waveform index in total sequencer
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
        w_index = np.arange(0, self.integration_length , 1)
        
        for channel,freq in enumerate(self.qubit_frequency):
            # assign real and image integration coefficient 
            # integration settings for one I/Q pair
            self.daq.setDouble('/{:s}/qas/0/integration/length'.format(self.id), self.integration_length)
            
            w_real = np.cos(w_index*freq/1e9/1.8*2*pi)
            w_imag = np.sin(w_index*freq/1e9/1.8*2*pi)

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
        
        get_path = lambda ch: '/{:s}/qas/0/result/data/{:d}/wave'.format(self.id,ch)
        chs = range(len(self.qubit_frequency))
        self.paths = list(map(get_path,chs))
        
        # for ch in range(len(self.qubit_frequency)):
            # path = '/{:s}/qas/0/result/data/{:d}/wave'.format(self.id,ch)
            # self.paths.append(path)

        if self.noisy:
            print('\n', 'Subscribed paths: \n ', self.paths, '\n')
        self.daq.subscribe(self.paths)

    # def stop_subscribe(self):       
    #     # Stop result unit
    #     self.daq.unsubscribe(self.paths)
    #     self.daq.setInt('/{:s}/qas/0/result/enable'.format(self.id), 0)

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
        
        get_vec = lambda v: v['vector']
        
        while time < timeout and not all(gotem.values()):
            if self.noisy:
                print('collecting results')
            dataset = daq.poll(poll_length, poll_timeout, poll_flags, poll_return_flat_dict)
            for p in paths:
                if p not in dataset:
                    continue
                chunks[p] = list(map(get_vec,dataset[p]))
                num_obtained = sum(map(len,chunks[p]))
                if num_obtained >= num_samples:
                    gotem[p] = True
                # for v in dataset[p]:
                    # chunks[p].append(v['vector'])
                    # num_obtained = sum([len(x) for x in chunks[p]])
                    # if num_obtained >= num_samples:
                        # gotem[p] = True
            time += poll_length
    
        if not all(gotem.values()):
            for p in paths:
                num_obtained = sum(map(len,chunks[p]))
                print('Path {}: Got {} of {} samples'.format(p, num_obtained, num_samples))
            raise Exception('Timeout Error: Did not get all results within {:.1f} s!'.format(timeout))
    
        # Return dict of flattened data
        return {p: np.concatenate(v) for p, v in chunks.items()}      

    def get_data(self):
        data = self.acquisition_poll(self.daq, self.paths, self.result_samples, timeout=10)
        return list(data.values())




@singletonMany
class zurich_hd:
    """server for zurich hd
    Args:
        obj_name (str): specify the object, the object with the same obj_name is singleton. 
        Example: If the class has been instantiated with obj_name=='obj1',  
        calling it again via 'zurich_hd(obj_name='obj1')' will not re-initiate but get the old object. 
    Instance: The instance now will return a dictionary of objects (value) and their obj_name (key) that has been created
    """
    def __init__(self,obj_name = 'HD_1',device_id='dev8334',labone_ip='localhost'):   
        self.id = device_id
        self.obj_name = obj_name
        self.noisy = False ## ## if open, will activate all print command during device working
        try:
            print('\nBring up %s in %s'%(self.id,labone_ip))
            self.daq = ziDAQ().daq
            self.daq.connectDevice(self.id,'1gbe')
            print(self.daq)
            self.FS = self.daq.getDouble('/{:s}/system/clocks/sampleclock/freq'.format(self.id)) ## sample rate
            self.init_setup()
        except:
            print('Failed to initialize, please check device.')


    def init_setup(self):
        self.waveform_length = [0,0,0,0] ## four awg's waveform length, unit --> Sample Number
        self.awg_grouping(grouping_index=0) ## index == 0 --> 4*2 grouping mode
        exp_setting = [
            ['/%s/sigouts/*/on'               % (self.id), 1], ## open all signal port
            ['/%s/sigouts/*/range'            % (self.id), 1], ## default output range: 1V
            ['/%s/awgs/0/outputs/*/amplitude' % (self.id), 1.0], ## hold 1 amplitude
            ['/%s/awgs/0/outputs/0/modulation/mode' % (self.id), 0], ## plain mode
            # ['/%s/system/awg/channelgrouping'% self.id, 0],  ## use 4*2 channels
            ['/%s/awgs/*/time'                 % self.id, 0], ## use maximum sample rate
            # ['/%s/awgs/0/userregs/0'           % self.id, 0],
            # ['/%s/system/clocks/sampleclock/freq' % self.id, self.FS],
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
        self.daq.setInt('/{:s}/awgs/0/auxtriggers/0/slope'.format(self.id), 1) ## digital trigger 1 slope: rise
        self.daq.setInt('/{:s}/awgs/0/auxtriggers/0/channel'.format(self.id), 0) ## DigTrigger --> trigger in 1
        self.daq.setInt('/{:s}/triggers/in/*/imp50'.format(self.id), 0)  ## DIO trigger in, set impedance
        self.daq.setDouble('/{:s}/triggers/in/*/level'.format(self.id), 0.4) ### trigger threshold level
        self.daq.setInt('/{:s}/awgs/0/single'.format(self.id), 1) ## close rerun
        # # Ensure that all settings have taken effect on the device before continuing.
        # self.daq.sync()
        print('%s: Complete Initialization'%self.id.upper())

    def reset_zi(self):
        print('关机重启吧！！')



    #####------- set & get HD parameter ------- #####
    def awg_open(self,awgs_index=[0,1,2,3]):
        ## run specific AWG following awgs_index
        for i in awgs_index:
            self.daq.setInt('/{:s}/awgs/{:d}/enable'.format(self.id, i), 1)
            if self.noisy:
                print('%s AWG%d running.'%(self.id,i))

    def awg_close(self,awgs_index=[0,1,2,3]):
        ## stop specific awg following awgs_index
        for i in awgs_index:
            self.daq.setInt('/{:s}/awgs/{:d}/enable'.format(self.id,i), 0)

    def awg_grouping(self,grouping_index=0):
        """ 
            grouping_index:
                0 : 4x2 with HDAWG8; 2x2 with HDAWG4.
                1: 2x4 with HDAWG8; 1x4 with HDAWG4.
                2 : 1x8 with HDAWG8.
            set AWG grouping mode, following path:
            '/dev_id/system/awg/channelgrouping', Configure 
            how many independent sequencers, should run on 
            the AWG and how the outputs are grouped by sequencer.
        """
        grouping = ['4x2 with HDAWG8', '2x4 with HDAWG8', '1x8 with HDAWG8']
        self.daq.setInt('/{:s}/system/awg/channelgrouping'.format(self.id), grouping_index)
        if grouping_index == 0:
            for awg in range(4):
                self.daq.setInt('/{:s}/awgs/{:d}/auxtriggers/0/channel'.format(self.id,awg), int(awg*2)) ## set digital trigger
                self.daq.setDouble('/{:s}/triggers/in/{:d}/level'.format(self.id,int(awg*2)), 0.1) ## set trigger threshold
        if grouping_index == 1:
            for awg in range(2):
                self.daq.setInt('/{:s}/awgs/{:d}/auxtriggers/0/channel'.format(self.id,awg), int(awg*4)) ## set digital trigger
                self.daq.setDouble('/{:s}/triggers/in/{:d}/level'.format(self.id,int(awg*4)), 0.2) ## set trigger threshold
        if grouping_index == 2:
            self.daq.setInt('/{:s}/awgs/{:d}/auxtriggers/0/channel'.format(self.id,0),0) ## set digital trigger
            self.daq.setDouble('/{:s}/triggers/in/{:d}/level'.format(self.id,0), 0.4) ## set trigger threshold
        self.daq.sync()
        print(' --> [%s] channel grouping: %s'%(self.id.upper(),grouping[grouping_index]))
        
    def update_pulse_length(self,awg_index):
        hdinfo = self.daq.getList('/{:s}/awgs/{:d}/waveform/waves/0'.format(self.id,awg_index))
        if len(hdinfo)==0:
            self.waveform_length[awg_index] = -1
        elif len(hdinfo)==1:
            self.waveform_length[awg_index] = int(len(hdinfo[0][1][0]['vector'])/2) ## consider two channel wave;
        else:
            raise Exception('Unknown HD infomation:\n',hdinfo)
        if self.noisy:
            print('[%s-AWG%d] update_pulse_length: %r'%(self.id,awg_index,self.waveform_length))

    #####------- bulid and send AWGs ------- #####
    def awg_builder(self,waveform=[[0]],port=[],awg_index=0,loop=False):
        """ Build waveforms sequencer. Then compile and send it to devices.
        """
        # create waveform
        wave_len = len(waveform[0])
        define_str = 'wave wave0 = vect(_w_);\n'
        str0 = ''
        play_str = ''
        if len(port) == 0:
            port = np.arange(len(waveform))+1

        j = 0
        for i in port:
            wf = waveform[j]
            str0 = str0 + define_str.replace('0', str(i)).replace('_w_', ','.join([str(x) for x in wf]))
            play_str = play_str + (','+ str(i) + ',wave'+str(i))
            j += 1
        play_str = play_str[1:]
        
        awg_program = textwrap.dedent("""\
        const f_s = _c0_;
        $str0
        while(1){
        waitDigTrigger(1);
        playWave($play_str);
        waitWave();
        }
        """)
        awg_program = awg_program.replace('$str0', str0)
        awg_program = awg_program.replace('$play_str', play_str)
        awg_program = awg_program.replace('_c0_', str(self.FS))
        if loop:
            awg_program = awg_program.replace('waitDigTrigger(1);', '//waitDigTrigger(1);')
        self.awg_upload_string(awg_program,awg_index=awg_index)
        self.update_pulse_length(awg_index=awg_index)

    def awg_upload_string(self,awg_program,awg_index=0): 
        """ awg_program: waveforms sequencer text
            awg_index: this device's awgs sequencer index. 
                       If awgs grouping == 4*2, this index 
                       can be selected as 0,1,2,3
            write into waveforms sequencer and compile it.
        """
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

    def reload_waveform(self,waveform,awg_index=0,index=0):
        """ waveform: (numpy.array) one/two waves with unit amplitude.
            awg_index: this devices awg sequencer index
            index: this waveform index in total sequencer
        """
        waveform_native = convert_awg_waveform(waveform)
        if self.noisy:
            print('[%s-AWG%d] reload waveform length: %d'%(self.id,awg_index,len(waveform_native)))
        path = '/{:s}/awgs/{:d}/waveform/waves/{:d}'.format(self.id,awg_index,index)
        self.daq.setVector(path, waveform_native)
      
    def send_waveform(self,waveform=[[0],[0]],awg_index=0,port=[]):
        """ (for awgs grouping mode 4*2 case.)
            waveform: waveform list be sent
            awg_index: awg index need to reload waveform
            port: awg's port be used to reload
            Here judge which awgs or port will be used
            to reload. Fill zeros at the end of waveform 
            to match the prior waveform length or compile 
            sequencer again.
        """
        _n_ = self.waveform_length[awg_index] - len(waveform[0])
        if _n_ >= 0:
            if len(waveform) == 1: ## only one port be used, another should fill zero
                waveform_add = [[],[]] ## port = 1 or 2, (3-port) is another one.
                waveform_add[port[0]-1] = np.hstack((waveform[0],np.zeros(_n_)))
                waveform_add[int(3-port[0])-1] = np.zeros(self.waveform_length[awg_index])
                # print('fill zeros in port ',(int(3-port[0])))
            elif len(waveform) == 2: ## default port given in order
                waveform_add = [np.hstack((w,np.zeros(_n_))) for w in waveform]
            else:
                print('Error waveform list (len=%r), Empty or too many.'%len(waveform))
            try:
                self.reload_waveform(waveform_add,awg_index=awg_index)
            except:
                self.update_pulse_length(awg_index=awg_index)
                _n_ = self.waveform_length[awg_index] - len(waveform[0])
                if _n_ >= 0:
                    if len(waveform) == 1: ## only one port be used, another should fill zero
                        waveform_add = [[],[]] ## port = 1 or 2, 3-port is another one.
                        waveform_add[port[0]-1] = np.hstack((waveform[0],np.zeros(_n_)))
                        waveform_add[int(3-port[0])-1] = np.zeros(self.waveform_length[awg_index])
                    elif len(waveform) == 2: ## default port given in order
                        waveform_add = [np.hstack((w,np.zeros(_n_))) for w in waveform]
                    else:
                        print('Error waveform list (len=%r), Empty or too many.'%len(waveform))
                    self.reload_waveform(waveform_add,awg_index=awg_index)
                else:
                    print('Bulid %r-awg%d Sequencer(len=%r > %r)'%(self.id,awg_index,len(waveform[0]),self.waveform_length[awg_index]))
                    if len(waveform) == 1:
                        waveform_add = [[],[]] ## port = 1 or 2, (3-port) is another one.
                        waveform_add[port[0]-1] = waveform[0]
                        waveform_add[int(3-port[0])-1] = np.zeros(len(waveform[0]))
                    else:
                        waveform_add = waveform
                    t0 = time.time()
                    self.awg_builder(waveform=waveform_add,port=port,awg_index=awg_index)
                    print('%r-awg%d builder: %.3f s'%(self.id,awg_index,time.time()-t0))
        else:
            print('Bulid [%s-AWG%d] Sequencer2 (len=%r > %r)'%(self.id,awg_index,len(waveform[0]),self.waveform_length[awg_index]))
            if len(waveform) == 1:
                waveform_add = [[],[]] ## port = 1 or 2, (3-port) is another one.
                waveform_add[port[0]-1] = waveform[0]
                waveform_add[int(3-port[0])-1] = np.zeros(len(waveform[0]))
            else:
                waveform_add = waveform
            t0 = time.time()
            self.awg_builder(waveform=waveform_add,port=[1,2],awg_index=awg_index)
            print('[%s-AWG%d] builder: %.3f s'%(self.id,awg_index,time.time()-t0))







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
    NOTE: waveform reload needs each two channel one by one.
    """
    def uint16_waveform(wave): # Prepare waveforms
        wave = np.asarray(wave)
        if np.issubdtype(wave.dtype, np.floating):
            return np.asarray((np.power(2, 15) - 1) * wave, dtype=np.uint16)
        return np.asarray(wave, dtype=np.uint16)

    data_tuple = (uint16_waveform(wave_list[0]),uint16_waveform(wave_list[1]))
    return np.vstack(data_tuple).reshape((-2,), order='F')

    