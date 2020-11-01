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

from zilabrad.pyle.util import sweeptools as st

from zilabrad.instrument import waveforms
import zilabrad.instrument.qubitServer as qubitServer
from zilabrad.instrument.QubitContext import qubitContext
from zilabrad.instrument.QubitContext import update_session

from zilabrad import mp

import labrad
from labrad.units import Unit,Value

ar = st.RangeCreator()
_unitSpace = ('V','mV','us','ns','s','GHz','MHz','kHz','Hz','dBm','rad','None')
V, mV, us, ns,s, GHz, MHz,kHz,Hz, dBm, rad,_l  = [Unit(s) for s in _unitSpace]



def bringup_device():
    qctx = qubitContext()
    qctx.refresh()
    return       



if __name__ == '__main__':
    user = input("Enter user (default: hwh)") or "hwh"
    ss = update_session(user=user)
    do_bringup = input("Skip Bringup? (enter 0 for skip, default 1)") or 1
    if do_bringup != '0':
        bringup_device()
    from zilabrad.instrument import zurichHelper
    reload(zurichHelper)

