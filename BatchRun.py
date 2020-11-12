# -*- coding: utf-8 -*-
"""Batched commands for the daily running.

We always use in ipython3： run BatchRun
It will bringup the devices (object), and store some daily commands
"""


from importlib import reload
import numpy as np

from zilabrad import *
from labrad.units import Unit,Value

_unitSpace = ('V','mV','us','ns','s','GHz','MHz','kHz','Hz','dBm','rad','None')
V, mV, us, ns,s, GHz, MHz,kHz,Hz, dBm, rad,_l  = [Unit(s) for s in _unitSpace]
# some useful units

mp = multiplex
dp = dataProcess
# add some abbreviated to use

dh = dp.datahelp()


if __name__ == '__main__':
    sample = connect_ZI(reset=False)

