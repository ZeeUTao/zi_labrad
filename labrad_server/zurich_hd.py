# -*- coding: utf-8 -*-
"""
Helpers for Zurich Instruments
including classes for AWG devices, microwave sources
"""

"""
### BEGIN NODE INFO
[info]
name = zurich hd
version = 0.0.1
description = Basic python server.

[startup]
cmdline = %PYTHON% %FILE%
timeout = 20

[shutdown]
message = 987654321
timeout = 5
### END NODE INFO
"""

import logging
import zhinst.utils ## create API object
import textwrap ## to write sequencer's code
import time ## show total time in experiments

import numpy as np 
from numpy import pi


from labrad import types as T, util

from labrad.server import LabradServer, setting
from labrad.units import m, s
from labrad.util import hydrant
from labrad import util

from twisted.internet import defer, reactor
from twisted.internet.defer import inlineCallbacks, returnValue



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

_default_id = 'dev8334'
_message_1 = "Enter device id (default: %s)"%(_default_id)
device_id = input(_message_1) or _default_id


class zurich_hd(LabradServer):
    """
    zurich hd server, 
    Note: the parameter c is context, carried from LabradServer
        
    """
    name = 'Zurich HD_' + str(device_id)
    
    def initServer(self,labone_ip='localhost'):  
        
        self.id = device_id
        self.noisy = False ## if True, print working logs
        try:
            print('Bring up %s in %s'%(self.id,labone_ip))
            self.daq = zhinst.ziPython.ziDAQServer(labone_ip,8004,6)
            self.daq.connectDevice(self.id,'1gbe')
            print(self.daq)
            self.pulse_length_s = 0 ## waveform length; unit --> Sample Number
            self.FS = 1.8e9/(self.daq.getInt('/{:s}/awgs/0/time'.format(self.id))+1) ## 采样率
            self.init_setup()
        except:
            print('Reset Errpr, Please Check your device')
    
    @setting(100,'daq')
    def _daq(self,c):
        return self.daq
    
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
        
        
    @setting(11, 'Noisy',is_noisy=['w','_','*2v'])
    def Noisy(self,c,is_noisy=None):
        # 0 for no noisy, else noisy
        if is_noisy is None:
            print(self.noisy)
        else:
            self.noisy = is_noisy
        print(self.noisy)
        

    @setting(2, 'reset_zi')
    def reset_zi(self,c):
        print('Reset ZI！！')
    

    @setting(3, 'awg_builder', waveform=['*2v'],ports=['*w'],awg_index=['w'])
    def awg_builder(self,c,waveform=[[0.0]],ports=[],awg_index=0):
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
        print(awg_program)
        
        awgModule = self.daq.awgModule()
        awgModule.set('awgModule/device', self.id)
        awgModule.set('awgModule/index', awg_index)# AWG 0, 1, 2, 3
        awgModule.execute()
        awgModule.set('awgModule/compiler/sourcestring', awg_program)
        
        # Ensure that compilation was successful
        while awgModule.getInt('awgModule/compiler/status') == -1:
            time.sleep(0.1)
        
        # compilation failed, raise an exception
        if awgModule.getInt('awgModule/compiler/status') == 1:
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
    
    @setting(5, 'reload_waveform', waveform=['*2v'],awg_index=['v'],index=['v'])
    def reload_waveform(self,c,waveform,awg_index=0,index=0): ## (dev_id, awg编号(看UI的awg core), index是wave的编号;)
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
            
    @setting(7, 'send_waveform',waveform=['*2v'],ports=['*s'],check=['b'])   
    def send_waveform(self,c,waveform=[[0],[0]],ports=[],check=False):
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

    @setting(8, 'awg_open')  
    def awg_open(self):
        """打开AWG， 打开输出，设置输出量程"""
        self.daq.setInt('/{:s}/awgs/{}/enable'.format(self.id, 0), 1)
        if self.noisy:
            print('\n AWG running. \n')
            
    @setting(9, 'awg_close_all')  
    def awg_close_all(self):
        self.daq.setInt('/{:s}/awgs/*/enable'.format(self.id), 0)

    @setting(10, 'awg_grouping', grouping_index = ['v'])  
    def awg_grouping(self,c, grouping_index = 2):
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
            



__server__ = zurich_hd()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
