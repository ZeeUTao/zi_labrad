# -*- coding: utf-8 -*-
"""Batched commands for the daily running.

We always use in ipython3： run BatchRun
It will bringup the devices (object), and store in locals()

To reload mp, you can call 'reload_mp', which fed 'devices' to mp.exp_devices
"""



from importlib import reload
import configparser
import os
import numpy as np
from zilabrad.pyle.registry import RegistryWrapper
from zilabrad.instrument import zurichHelper as zH
from zilabrad.pyle.workflow import switchSession
from zilabrad.pyle.util import sweeptools as st

from zilabrad.instrument import waveforms
import zilabrad.instrument.qubitServer as qubitServer

import labrad
from zilabrad.pyle.units import Unit,Value

from qcodes.instrument.channel import ChannelList, InstrumentChannel

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

def bringup_device(modes):
    dev = loadInfo(paths=['Servers','devices']) ## only read
    for m in modes: 
        if m == 1:
            qa = zH.zurich_qa(dev.zi_qa_id)
            devices[m-1] = qa
        if m == 2:
            hd = zH.zurich_hd(dev.zi_hd_id)
            devices[m-1] = hd
        if m == 3:
            mw = zH.microwave_source(dev.microwave_source_xy,'mw')
            devices[m-1] = mw
        if m == 4:
            mw_r = zH.microwave_source(dev.microwave_source_readout,'mw_r')
            devices[m-1] = mw_r
        if m == 5:
            wfs = waveforms.waveform()
            devices[m-1] = wfs
    return       



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



mpreload = r'reload(mp);mp.exp_devices=devices'
"""
Quick commands:

exec(mpreload)
#quick eval command, example:

devices[0].noisy = True
#devices will print some strings about running
"""
from zilabrad import *

_default_modes = [1,2,3,4,5]



if __name__ == '__main__':
    user = input("Enter user (default: hwh)") or "hwh"
    ss = update_session(user=user)


    do_bringup = input("Skip Bringup? (enter 0 for skip, default 1)") or 1
    if do_bringup != '0':
        devices = [None]*len(_default_modes)
        modes = _default_modes
        
        bringup_device(modes=modes)
        mp.exp_devices = devices
        for dev in mp.exp_devices:
            print(dev.__class__.__name__)
    from zilabrad.instrument import zurichHelper
    reload(zurichHelper)

