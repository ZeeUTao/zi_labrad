# -*- coding: utf-8 -*-
"""Batched commands for the daily running.

We always use in ipython3： run BatchRun
It will bringup the devices (object), and store in locals()

To reload mp, you can call 'reload_mp', which fed 'devices' to mp.exp_devices
"""



from importlib import reload
import configparser
import os

from zilabrad.pyle.registry import RegistryWrapper
from zilabrad.instrument import zurichHelper as zH
from zilabrad.pyle.workflow import switchSession
from zilabrad.pyle.util import sweeptools as st

import labrad
from labrad.units import Unit,Value


ar = st.RangeCreator()
_unitSpace = ('V','mV','us','ns','s','GHz','MHz','kHz','Hz','dBm','rad','None')
V, mV, us, ns,s, GHz, MHz,kHz,Hz, dBm, rad,_l  = [Unit(s) for s in _unitSpace]


def loadInfo(paths=['Servers','devices']):
    """
    load the sample information from specified directory.

    Args:
        paths (list): Array with data of waveform 1.

    Returns: 
        reg.copy() (dict): 
        the key-value information from the directory of paths

    ** waveform reload needs each two channel one by one.
    """
    cxn=labrad.connect()
    reg = RegistryWrapper(cxn, ['']+paths)
    return reg.copy()

def bringup_device(modes=[1,2,3,4]):
    dev = loadInfo(paths=['Servers','devices']) ## only read
    for m in modes:
        if m == 1:
            qa = zH.zurich_qa(dev.zi_qa_id)
        if m == 2:
            hd = zH.zurich_hd(dev.zi_hd_id)
        if m == 3:
            mw = zH.microwave_source(dev.microwave_source_xy,'mw')
        if m == 4:
            mw_r = zH.microwave_source(dev.microwave_source_readout,'mw_r')      
    return qa,hd,mw,mw_r        
            
"""Instance: physical units

defined via Unit(class) from labrad.units
"""



"""Instance: Simple generator that encapsulates a range of values with units

Example：
freq = ar[1.0:3.0:0.01,GHz] 
"""
def update_session(user='hwh'):
    cxn=labrad.connect()
    ss = switchSession(cxn,user=user,session=None)

    # curpath = os.path.dirname(os.path.realpath(__file__))
    # cfgpath = os.path.join(curpath, "zi_config.ini")
     
    # conf = configparser.ConfigParser()
    # conf.read(cfgpath, encoding="utf-8")
     
    # conf.set("config", "_default_session", str(ss._dir)) 
    # with open(cfgpath, "r+", encoding="utf-8") as file:
        # conf.write(file)
    return ss



def reload_mp():
    reload(mp)
    mp.exp_devices = devices
    
_default_modes = [1,2,3,4]


user = input("Enter user (default: hwh)") or "hwh"
ss = update_session(user=user)

do_bringup = input("Skip Bringup? (enter 0 for skip, default 1)") or 1
if do_bringup != '0':
    modes = _default_modes
    import zilabrad.mp as mp
    devices = bringup_device(modes=modes)
    mp.exp_devices = devices
    print(mp.exp_devices)
    

from zilabrad.instrument import zurichHelper


reload(zurichHelper)

