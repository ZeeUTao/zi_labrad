# -*- coding: utf-8 -*-
"""
Resources for qubit and devices
"""
import importlib ## for import other class by string


from zilabrad.pyle.workflow import switchSession
from zilabrad.pyle.registry import RegistryWrapper
from zilabrad.pyle.tools import singleton


import labrad



@singleton
class qubitContext(object):
    """
    resources of the experimental devices, 
    use get_server(device name) to get 
    active device instance.
    """

    def __init__(self,cxn=None,user_sample=None):
        """
        Args:
            servers (dict): devices dictionary
        """
        if cxn is None:
            self.cxn = labrad.connect()
        else:
            self.cxn = cxn
            
        if user_sample is not None:
            self.user_sample = user_sample
        self._ini_setup()

    def _ini_setup(self):
        self.servers = {}
        self.servers_group = {'ArbitraryWaveGenerator':{},
                              'MicrowaveSource':{},
                              'DataAcquisition':{},
                              'QuantumAnalyzer':{},
                              'DCsource':{},
                              'Oscilloscope':{},
                              'SpectrumAnalyzer':{},
                              'VectorNetworkAnalyzer':{},
                              'BasicDriver':{},
                              'Virtual':{},
                             }

        self.deviceInfo = RegistryWrapper(self.cxn,['','Servers', 'Devices']).copy()
        self.ActiveDevices = self.deviceInfo['ActiveDevices']

    def _active_all_devices(self):
        for dev_name in self.deviceInfo['ActiveDevices']:
            dev = self.get_server(name=dev_name,active=True,raw=False)

        if 'ziDAQ' in self.servers.keys(): ## 控制苏黎世仪器的接口
            self.daq = self.servers['ziDAQ'].daq


    def get_server(self, name: str or None, active=False, raw=False):
        """return the object of server
        example: get_server('qa_1')
        """
        if active:
            if name in self.deviceInfo.keys():
                driver_path = self.deviceInfo[name]['driver_path']
                device_class = self.deviceInfo[name]['driver_class']
                address = self.deviceInfo[name]['address']
                ## Load the device driver python file
                try:
                    device_driver = importlib.import_module(driver_path)
                    dev = getattr(device_driver,device_class)(name,address)
                except Exception as e:
                    print('[%s] create failed:\n    path: %s \n    class: %s \n    address: %s'%
                        (name,driver_path,device_class,address))
                    raise e
                ## 将单例类的实例从dict中取出
                if isinstance(dev,dict) and raw == False: 
                    dev = dev[name]
                self.servers[name] = dev
                self.servers_group[self.deviceInfo[name]['device_type']][name] = dev
                print('connected [%s] --> %s'%(name,address))
                return dev
            else:
                print('Cannot find Server [%s]!'%name)
                return
        else:
            if name in self.servers.keys():
                return self.servers[name]
            elif name in self.deviceInfo.keys():
                print('[%s] not activated!'%name)
                return
            else:
                print('Cannot find Server [%s]!'%name)
                return

    def get_servers_group(self, name: str):
        """return a collection of servers for the same type
        (if they have more than ones)
        """
        if name in self.servers_group.keys():
            return self.servers_group[name]
        else:
            print('Unknown device type [%s]!'%name)
            return

    def get_type(self,name:str):
        if name in self.deviceInfo.keys():
            return self.deviceInfo[name]['device_type']
        else:
            print('Cannot find Server [%s]!'%name)
            return

    # def remove_server(self,name): ## 单例怎么删啊!!
    #     if name in self.deviceInfo.keys():
    #         dev_dict = self.get_server(name=name,active=True,raw=True)
    #         del dev_dict[name]
    #     else:
    #         print('Cannot find Server [%s]!'%name)
    #         return


    def refresh(self):
        self._ini_setup()
        self._active_all_devices()











