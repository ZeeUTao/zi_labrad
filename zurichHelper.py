import zhinst.utils ## create API object
import textwrap ## to write sequencer's code
import time ## show total time in experiments
import matplotlib.pyplot as plt ## give picture
import numpy as np 
from numpy import pi
from math import ceil
from pyle.registry import RegistryWrapper
import labrad
from labrad.units import Unit,Value
_unitSpace = ('V', 'mV', 'us', 'ns','s', 'GHz', 'MHz','kHz','Hz', 'dBm', 'rad','None')
V, mV, us, ns,s, GHz, MHz,kHz,Hz, dBm, rad,_l  = [Unit(s) for s in _unitSpace]

# init
qa,hd,mw,yoko = [None]*4


def loadInfo(paths=['Servers','devices']):
    cxn=labrad.connect()
    reg = RegistryWrapper(cxn, ['']+paths)
    return reg.copy()

def bringup_device(modes=[1,2]):
    global qa,hd,mw,yoko
    dev = loadInfo(paths=['Servers','devices']) ## only read
    for m in modes:
        if m == 1:
            qa = zurich_qa(dev.zi_qa_id)
        if m == 2:
            hd = zurich_hd(dev.zi_hd_id)
        if m == 3:
            mw = microwave_source(dev.microwave_source_ip)
        # if m == 4:
        #     yoko = yokogawa_dc_source(dev.yokogawa_dc_ip)

def init_device(back=False):
    """
    initialize all device setup; 
    """
    qa.init_setup()
    hd.init_setup()
    # mw.init_setup()
    # yoko.init_setup()

def stop_device():
    qa.stop_subscribe()
    hd.awg_close_all()
    # dv.close(ctx)
    # mw.set_output(False)
    # yoko.set_off()


def mpAwg_init(q,stats):
    ## stats: Number of Samples for one sweep point;
    ## get commonly used value
    sb_freq = q['readout_freq'][Hz] - q['readout_fc'][Hz]
    port_xy,port_dc,port_z = int(q['chan_xy'][2]),int(q['chan_dc'][2]),int(q['chan_z'][2])

    ## initialize instruments
    # t0 = time.time()
    # init_device() ## 初始化仪器的设置,避免上次实验的影响; 这部分用时较长；
    # print('initialized time:',time.time()-t0)


    t0 = time.time()
    hd.pulse_length_s = 0
    qa.result_samples = stats  ## sample number for one sweep point
    qa.set_adc_trig_delay(q['bias_start'][s]+hd.pulse_length_s+q['readout_delay'][s])
    qa.set_pulse_length(q['readout_len'][s])
    qa.set_qubit_frequency([sb_freq])
    qa.set_subscribe(source=7)
    print('qa-set_para:',time.time()-t0)
    # mw.set_power(q['microwave_power']['dBm'])
    # mw.set_freq(q['readout_fc'][Hz])

    ## initialize zi's awgs and building 
    w_qa = waveform(all_length=q['bias_start'][s]+hd.pulse_length_s+qa.pulse_length_s+q['bias_end'][s],fs=1.8e9,origin=-q['bias_start'][s])
    w_hd = waveform(all_length=q['bias_start'][s]+hd.pulse_length_s+qa.pulse_length_s+q['bias_end'][s],fs=2.4e9,origin=-q['bias_start'][s])
    q_dc = w_hd.square(amp=0)
    # q_r = [w_qa.square(amp=0),w_qa.square(amp=0)] ## (I,Q)--Two Channels
    q_r = [w_qa.square(amp=0.00,start=0,end=0+q['readout_len'][s]),w_qa.square(amp=0.00,start=0,end=0+q['readout_len'][s])]
    # print(len(q_dc),len(q_r[0]))

    ## set hd's awgs waveforms
    t0 = time.time()
    hd.awg_builder(waveform = [q_dc], ports=[port_dc], trigger=1) ## 开启hd等待trigger的模式
    print('hd-awg_builder:',time.time()-t0)
    ## set qa's awgs waveforms
    t0 = time.time()
    qa.awg_builder(waveform = q_r)
    print('qa-awg_builder:',time.time()-t0)
    ### ----- finish ----- ###
    return w_qa,w_hd


class zurich_qa(object):
    def __init__(self,device_id):   
        self.id = device_id
        self.FS = 1.8e9 ## 采样率
        self.noisy = False ## 开启的话, 会打开所有正常工作时的print语句
        required_devtype = 'UHFQA'
        required_options = ['']
        try:
            self.daq, self.device,info= zhinst.utils.create_api_session(device_id, 6,required_devtype=required_devtype,required_options=required_options)
            print(self.daq)
            zhinst.utils.disable_everything(self.daq, self.device)       
        except:
            print('初始化失败，请检查仪器')
        self.init_setup()

    def init_setup(self):
        self.average = 1 #不用硬件平均。保持默认不变
        self.result_samples = 100
        self.channels = [] # channel from 1 to 10;
        self.qubit_frequency = [] # match channel number
        self.paths = [] # save result path 
        self.source = 7 ## 解调模式, 7=integration; 

        self.set_adc_trig_delay(0) ## delay after trigger; unit -> second
        self.set_pulse_length(0)   ## length of qa's awg, unit --> second

        self.daq.setInt('/{:s}/qas/0/integration/mode'.format(self.device), 0) ## set standard mode to integration
        self.daq.setDouble('/{:s}/qas/0/delay'.format(self.device), 0); # delay 0 samples for integration
        # 跳过crosstalk 操作，节约时间
        self.daq.setInt('/{:s}/qas/0/crosstalk/bypass'.format(self.device), 1)
        # set deskew matrix  
        self.daq.setDouble('/{:s}/qas/0/deskew/rows/0/cols/0'.format(self.device), 1)
        self.daq.setDouble('/{:s}/qas/0/deskew/rows/1/cols/1'.format(self.device), 1)
        self.daq.setDouble('/{:s}/qas/0/deskew/rows/1/cols/0'.format(self.device), 1)
        self.daq.setDouble('/{:s}/qas/0/deskew/rows/0/cols/1'.format(self.device), 1)
        #设置输出量程
        self.daq.setInt('/{:s}/awgs/0/outputs/*/mode'.format(self.device),0) ## output mode: plain
        self.daq.setDouble('/{:s}/sigouts/*/range'.format(self.device), 1) # output 1.0 V_peak
        #设置输入量
        self.daq.setDouble('/{:s}/sigins/*/range'.format(self.device), 1) # 输入量程1V
        self.daq.setInt('/{:s}/sigins/*/imp50'.format(self.device), 1) # 50 Ohm 输入阻抗
        
        self.daq.setInt('/{:s}/triggers/out/0/source'.format(self.device), 32) ## open trig 1
        self.daq.setInt('/{:s}/triggers/out/1/source'.format(self.device), 33) ## open trig 2
        self.daq.setInt('/{:s}/triggers/out/*/drive'.format(self.device), 1)
        # 设置DIO输出为qubit result
        self.daq.setInt('/{:s}/dios/0/drive'.format(self.device), 15)
        self.daq.setInt('/{:s}/dios/0/mode'.format(self.device), 2)
        # Sample DIO data at 50 MHz
        self.daq.setInt('/{:s}/dios/0/extclk'.format(self.device), 2)

        # self.daq.setInt('/{:s}/qas/0/result/length'.format(self.device), self.result_samples) #results length 
        # self.daq.setInt('/{:s}/qas/0/result/averages'.format(self.device), self.average) # average results
        #以下会设置读取脉冲的参数
        # self.daq.setDouble('/{:s}/awgs/0/userregs/0'.format(self.device), self.average)# average results
        # self.daq.setDouble('/{:s}/awgs/0/userregs/1'.format(self.device), self.result_samples)# #results length
        """打开AWG， 打开输出，设置输出量程"""
        # run = 1 start AWG, run =0 close AWG
        self.daq.setInt('/{:s}/sigouts/*/on'.format(self.device), 1)
        self.daq.setInt('/{:s}/awgs/0/single'.format(self.device), 1)
        # Arm the device
        self.daq.sync() ## 同步设置

    # def set_result_samples(self,num):
    #     self.result_samples = num
    #     self.daq.setDouble('/{:s}/awgs/0/userregs/1'.format(self.device), self.result_samples)# #results length

    ####--AWG波形设置部分--####
    def set_adc_trig_delay(self,delay):
        self.adc_trig_delay_s = int(delay*self.FS/8)*8/self.FS ## unit: second
        self._adc_trig_delay_ = self.adc_trig_delay_s*self.FS ## delay from dc trigger to ADC trigger; unit -> Sample Number

    def set_pulse_length(self, length):
        self.pulse_length_s = int(length*self.FS/8)*8/self.FS
        self._pulse_length_ =  self.pulse_length_s*self.FS ## unit --> Sample Number
        self._integration_length_ = self._pulse_length_ ## unit --> Sample Number

    def awg_builder(self, sampling_rate = 1.8e9, waveform = [[0.0], [0.0]], trigger = 0):
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
            play_str = play_str + (  ','+ str(i+1) + ', wave'+str(i+1) )
        
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
                playWave($play_str);
                wait(_c1_); // wait time = qa._adc_trig_delay_
                setTrigger(AWG_INTEGRATION_ARM + AWG_INTEGRATION_TRIGGER + AWG_MONITOR_TRIGGER);// 开始抓取信号，进行integration
                setTrigger(AWG_INTEGRATION_ARM);//重置
                waitWave();
                wait(200e-6*f_s/8); //等待200us后发送下一个
        }
        //wait(200e-6*f_s/8); // 间隔200us
        }        
        """)
        awg_program = awg_program.replace('_c0_', str(sampling_rate))
        awg_program = awg_program.replace('_c1_', str(self._adc_trig_delay_/8))
        awg_program = awg_program.replace('_c2_', str(self.result_samples))
        awg_program = awg_program.replace('_c3_', str(self.average))
        awg_program = awg_program.replace('$str0', str0)
        awg_program = awg_program.replace('$play_str', play_str)

        # awg_program = textwrap.dedent("""\
        # const FS = 1.8e9;
        # const LENGTH = _c1_;
        # const N = floor(LENGTH*FS);
        # wave w = ones(N);

        # setTrigger(AWG_INTEGRATION_ARM);

        # repeat(_c2_) {
        #     playWave(w, w);
        #     //wait(0);
        #     setTrigger(AWG_INTEGRATION_ARM + AWG_INTEGRATION_TRIGGER +AWG_MONITOR_TRIGGER);
        #     setTrigger(AWG_INTEGRATION_ARM);
        #     waitWave();
        #     wait(200e-6*FS/8);
        # }
        # """)
        # awg_program = awg_program.replace('_c1_', str(self.pulse_length_s))
        # awg_program = awg_program.replace('_c2_', str(self.result_samples))

        # #如果是触发模式，加入触发指令，触发在trigger input1 的上升沿
        # if trigger !=0:
        #     awg_program = awg_program.replace('//waitDigTrigger(1);','waitDigTrigger(1);') # set trigger
        #     awg_program = awg_program.replace('setTrigger(0b11);','//setTrigger(0b11);') # set trigger
        #     # trigger on the rise edge, the physical trigger port is trigger 1 at the frnont panel
        #     #impedance 50 at trigger input. trigger level 0.7 
        #     self.daq.setInt('/{:s}/awgs/0/auxtriggers/0/slope'.format(self.device), 1)
        #     self.daq.setInt('/{:s}/awgs/0/auxtriggers/0/channel'.format(self.device), 0)
        #     self.daq.setInt('/{:s}/triggers/in/0/imp50'.format(self.device), 0)            
        #     self.daq.setDouble('/{:s}/triggers/in/0/level'.format(self.device), 0.7)
        self.awg_upload_string(awg_program)

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


    def reload_waveform(self,waveform,awg_index=0,index=0): ## (dev_id, awg编号(默认0), index是wave编号;)
        waveform_native = zhinst.utils.convert_awg_waveform(waveform[0],waveform[1])
        path = '/{:s}/awgs/{:d}/waveform/waves/{:d}'.format(self.id,awg_index,index)
        # print(len(waveform_native))
        self.daq.setVector(path, waveform_native)


    def awg_open(self):
        self.daq.syncSetInt('/{:s}/awgs/0/enable'.format(self.device), 1)
        if self.noisy:
            print('\n AWG running. \n')


    ####--设置比特的读取频率sideband部分--####
    def set_qubit_frequency(self, frequency_array):
        #设置解调用的weights, 更新 channels, 和paths
        self.channels = [] # channel from 0 to 9;
        self.qubit_frequency = frequency_array
        for i in range(len(self.qubit_frequency)):
            self.channels.append(i)  #更新解调器/通道   
        self.set_all_integration()
        self.set_subscribe()

    def set_all_integration(self):
        for i in range(len(self.qubit_frequency)):
            self.set_qubit_integration_I_Q(i,self.qubit_frequency[i])

    def set_qubit_integration_I_Q(self, channel, qubit_frequency): 
        # assign real and image integration coefficient 
        # integration settings for one I/Q pair
        from numpy import pi
        self.daq.setDouble('/{:s}/qas/0/integration/length'.format(self.device), self._integration_length_)
        w_index      = np.arange(0, self._pulse_length_ , 1)
        weights_real = np.cos(w_index/1.8e9*qubit_frequency*2*pi)
        weights_imag = np.sin(w_index/1.8e9*qubit_frequency*2*pi)
        w_real = np.array(weights_real)
        w_imag = np.array(weights_imag)
        self.daq.setVector('/{:s}/qas/0/integration/weights/{}/real'.format(self.device, channel), w_real)
        self.daq.setVector('/{:s}/qas/0/integration/weights/{}/imag'.format(self.device, channel), w_imag)
        # set signal input mapping for QA channel : 0 -> 1 real, 2 imag
        self.daq.setInt('/{:s}/qas/0/integration/sources/{:d}'.format(self.device, channel), 0)


    ####--读取数据部分--####
    def acquisition_poll(self, daq, paths, num_samples, timeout=10.0):
        """ Polls the UHFQA for data.
    
        Args:
            paths (list): list of subscribed paths
            num_samples (int): expected number of samples
            timeout (float): time in seconds before timeout Error is raised.
        """
        if self.noisy:
            print('collecting results')
        poll_length = 0.001  # s
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
        self.daq.setInt('/{:s}/qas/0/result/enable'.format(self.device), 0)

    def set_subscribe(self,source=None): 
        """配置采集数据所需设置,设定读取result的path
        """
        if source==None:
            source = self.source
        self.paths = []
        self.daq.setInt('/{:s}/qas/0/result/length'.format(self.device), self.result_samples)#results length 
        self.daq.setInt('/{:s}/qas/0/result/averages'.format(self.device), self.average)# average results
        
        self.daq.setInt('/{:s}/qas/0/result/reset'.format(self.device), 1) ## reset
        self.daq.setInt('/{:s}/qas/0/result/enable'.format(self.device), 1) ##启动qa_result模块,等待数据
        self.daq.sync() ## 同步设置
        self.daq.setInt('/{:s}/qas/0/result/source'.format(self.device), source) # integration=7
        if self.noisy:
            print(self.channels)
        for ch in self.channels:
            path = '/{:s}/qas/0/result/data/{:d}/wave'.format(self.device,ch)
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
    def __init__(self,device_id):
        self.id = device_id
        self.FS = 2.4e9 ## 采样率
        self.pulse_length_s = 0 ## waveform length; unit --> Sample Number
        self.noisy = False
        """初始化AWG，仪器都为默认设置"""
        required_devtype = 'HDAWG'
        required_options = ['']
        try:
            self.daq, self.device,info= zhinst.utils.create_api_session(device_id, 6,required_devtype=required_devtype,required_options=required_options)
            zhinst.utils.disable_everything(self.daq, self.device)       
        except:
            print('初始化失败，请检查仪器')

        self.init_setup()

    def init_setup(self):
        self.pulse_length_s = 0
        amplitude = 1.0
        exp_setting = [
            ['/%s/sigouts/*/on'               % (self.device), 1],
            ['/%s/sigouts/*/range'            % (self.device), 1],
            ['/%s/awgs/0/outputs/*/amplitude' % (self.device), amplitude],
            ['/%s/awgs/0/outputs/0/modulation/mode' % (self.device), 0],
            ['/%s/system/awg/channelgrouping'% self.device, 2],  ## use 1*8 channels
            ['/%s/awgs/0/time'                 % self.device, 0],
            ['/%s/awgs/0/userregs/0'           % self.device, 0],
            ['/%s/system/clocks/sampleclock/freq' % self.device, self.FS],
            ['/%s/system/clocks/referenceclock/source' % self.device, 1], ## set ref clock mode in 'External'
            ['/%s/awgs/0/dio/strobe/slope' % self.device, 0],
            ['/%s/awgs/0/dio/strobe/index' % self.device, 15],
            ['/%s/awgs/0/dio/valid/index' % self.device, 0],
            ['/%s/awgs/0/dio/valid/polarity' % self.device, 2],
            ['/%s/awgs/0/dio/mask/value' % self.device, 7], # 111 三位qubit results
            ['/%s/awgs/0/dio/mask/shift' % self.device, 1],
            ['/%s/raw/dios/0/extclk' % self.device, 2]
        ]
        self.daq.set(exp_setting)
        self.daq.setInt('/{:s}/awgs/0/auxtriggers/0/slope'.format(self.device), 1)
        self.daq.setInt('/{:s}/awgs/0/auxtriggers/0/channel'.format(self.device), 0)
        self.daq.setInt('/{:s}/triggers/in/0/imp50'.format(self.device), 0)  ## ???
        self.daq.setDouble('/{:s}/triggers/in/0/level'.format(self.device), 0.7) ### ???
        self.daq.setInt('/{:s}/awgs/0/single'.format(self.device), 1)
        # Ensure that all settings have taken effect on the device before continuing.
        self.daq.sync()

    def awg_builder(self, waveform = [[0.11,0.222], [0.333,0.444]],ports=[], trigger = 1, awg_index = 0):
        """根据波形构建一个awg 程序， waveform的波形同时播放，可以工作在被触发模式trigger =1 """
        #sampling rate = 2.4e9
        #waveform = [wave1, wave2, wave3,...]
        #trigger =1 tgrigger mode on; trigger = 0 tigger removed.
        
        #用来构建简单的AWG sequence 程序，可以无限循环播放两个通道的两个波形
        #可以带触发，trigger = 1; 如果不打触发，waitDigTrigger(1) 会被注释掉
        #程序整体结构如下
        #wave wave1 = vect(0,0,0,0,1.0,1.0,1,1,0,0,0,0,1,1,1,1,0,0,0,0);
        #wave wave2 = vect(0,0,0,0,1.0,1.0,1,1,0,0,0,0,1,1,1,1,0,0,0,0);
        
        #while(1){
        #setTrigger(1);
        #setTrigger(0);
        #//waitDigTrigger(1)；
        #playWave(1, wave1,2, wave2);
        #}
        
        #检查波形每个点的值是否超过1、波形是没有单位的，不应该超过1
        # assert np.max(waveform) <= 1, '\n waveform value is dimensionless, less than 1. Check waveform values before proceed.\n'     
        #构建波形
        define_str = 'wave wave0 = vect(_w_);\n'
        str0 = ''
        play_str = ''
        if len(ports) == 0:
            ports = np.arange(len(waveform))+1
        j = 0
        for i in ports:
            # wave wave1 = vect(_w1); wave wave2 = vect(_w2_);...
            str0 = str0 + define_str.replace('0', str(i)).replace('_w_', ','.join([str(x) for x in waveform[j]]))
            # 1, w1, 2, w2, ....
            play_str = play_str + (  ','+ str(i) + ', wave'+str(i) )
            j += 1
        
        play_str = play_str[1:]
        
        awg_program = textwrap.dedent("""\
        $str0
        while(1){
        setTrigger(0b000);
        setTrigger(0b001);
        waitDigTrigger(1);
        playWave($play_str);
        waitWave();
        }
        """)
        awg_program = awg_program.replace('$str0', str0)
        awg_program = awg_program.replace('$play_str', play_str)
        
        # #如果是触发模式，加入触发指令，触发在trigger input1 的上升沿
        # if trigger !=0:
        #     awg_program = awg_program.replace('//waitDigTrigger(1)；','waitDigTrigger(1);') # set trigger
        #     # trigger on the rise edge, the physical trigger port is trigger 1 at the frnont panel
        #     # impedance 50 at trigger input. trigger level 0.7 
        #     self.daq.setInt('/{:s}/awgs/0/auxtriggers/0/slope'.format(self.device), 1)
        #     self.daq.setInt('/{:s}/awgs/0/auxtriggers/0/channel'.format(self.device), 0)
        #     self.daq.setInt('/{:s}/triggers/in/0/imp50'.format(self.device), 0)            
        #     self.daq.setDouble('/{:s}/triggers/in/0/level'.format(self.device), 0.7)
        
        # #如果需要一次播放多个信号，我们需要更改通道的grouping
        # if len(waveform)<3:
        #     self.awg_grouping(0)
        # if len(waveform)>2&len(waveform)<4:
        #     self.awg_close_all()
        #     self.awg_grouping(1)
        # if len(waveform)>4:
        #     self.awg_close_all()
        #     self.awg_grouping(2)

        self.awg_upload_string(awg_program, awg_index)


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

    def reload_waveform(self,waveform,awg_index=0,index=0): ## (dev_id, awg编号(默认0), index是wave的编号;)
        waveform_native = zhinst.utils.convert_awg_waveform(waveform)
        path = '/{:s}/awgs/{:d}/waveform/waves/{:d}'.format(self.id,awg_index,index)
        # print(len(waveform_native))
        self.daq.setVector(path, waveform_native)


    def awg_open(self):
        """打开AWG， 打开输出，设置输出量程"""
        # run = 1 start AWG, run =0 close AWG
        self.daq.setInt('/{:s}/awgs/{}/enable'.format(self.device, 0), 1)
        if self.noisy:
            print('\n AWG running. \n')

    def awg_close_all(self):
        self.daq.setInt('/{:s}/awgs/*/enable'.format(self.device), 0)


    def awg_grouping(self, grouping_index = 2):
        """配置AWG grouping 模式，确定有几个独立的序列编辑器。有多种组合"""
        #% 'system/awg/channelgrouping' : Configure how many independent sequencers
        #%   should run on the AWG and how the outputs are grouped by sequencer.
        #%   0 : 4x2 with HDAWG8; 2x2 with HDAWG4.
        #%   1: 2x4 with HDAWG8; 1x4 with HDAWG4.
        #%   2 : 1x8 with HDAWG8.
        grouping = ['4x2 with HDAWG8', '2x4 with HDAWG8', '1x8 with HDAWG8']
        self.daq.setInt('/{:s}/system/awg/channelgrouping'.format(self.device), grouping_index)
        self.daq.sync()
        if self.noisy:
            print('\n','HDAWG channel grouping:', grouping[grouping_index], '\n')
        



class microwave_source(object):
    def __init__(self,device_ip):
        self.ip = device_ip
        self.freq = 0
        self.power = 0
        rm = pyvisa.ResourceManager()
        self.dev = rm.open_resource('TCPIP0::%s::inst0::INSTR' %device_ip)
        self.init_setup()

    def init_setup(self):
        while self.get_output() == 0:
            self.set_output(1)

    def set_freq(self,freq):
        self.dev.write(':sour:freq %f Hz'%freq)

    def get_freq(self):
        self.freq = float(self.dev.query(':sour:freq?'))
        return self.freq

    def set_power(self,power):
        self.dev.write(':sour:pow %f'%power) ## unit:dBm

    def get_power(self):
        self.power = float(self.dev.query(':sour:pow?'))
        return self.power

    def get_output(self):
        state = self.dev.query(':outp:stat?')
        return int(state)

    def set_output(self,state):
        self.dev.write(':outp:stat %d'%state)



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




class waveform(object):
    """ 2020.09.17 hwh
    Create waveform's array from different device; 
    Consider total length and sampling frequency at first;
    """
    def __init__(self,all_length=1e-6,fs=1.8e9,origin=0):
        self.fs = fs
        self.sample_number = ceil(all_length*self.fs/8)*8
        self.len = self.sample_number/self.fs ## set as 8 sample integral multiple;
        self.origin = origin ## mark real start in waveform; set QA trigger as 0  
        self.tlist = np.asarray([k/self.fs+self.origin for k in range(self.sample_number)])

    def square(self,start=50e-9,end=150e-9,amp=1.0):
        pulse = [amp*(start<=t<end) for t in self.tlist]
        return np.asarray(pulse)

    def sine(self,amp=0.1,phase=0.0,start=0,end=1e-6,freq=10e6):
        pulse = [amp*np.sin(2*pi*freq*t+phase)*(start<=t<end) for t in self.tlist]
        return np.asarray(pulse)
    
    def cosine(self,amp=0.1,phase=0.0,start=0,end=1e-6,freq=10e6):
        pulse = [amp*np.cos(2*pi*freq*t+phase)*(start<=t<end) for t in self.tlist]
        return np.asarray(pulse)