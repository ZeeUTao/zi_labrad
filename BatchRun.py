# -*- coding: utf-8 -*-
"""Batched commands for the daily running.

We always open an 'ipython', and type 'run BatchRun' to start;
Then use the functions 'mp.xxx' (for example 'mp.s21_scan') to implement different experiments.
"""

import matplotlib.pyplot as plt
import numpy as np
from importlib import reload

import conf
import labrad
from pyle.workflow import switchSession
from pyle.util import sweeptools as st
from labrad.units import Unit,Value
_unitSpace = ('V', 'mV', 'us', 'ns','s', 'GHz', 'MHz','kHz','Hz', 'dBm', 'rad','None')
V, mV, us, ns,s, GHz, MHz,kHz,Hz, dBm, rad,_l  = [Unit(s) for s in _unitSpace]
"""Instance: physical units

defined via Unit(class) from labrad.units
"""

ar = st.RangeCreator()
"""Instance: Simple generator that encapsulates a range of values with units

Example：
freq = ar[1.0:3.0:0.01,GHz] 
"""

if __name__=="__main__":
	cxn=labrad.connect()
	ss = switchSession(cxn,user='hwh',session=None) 

	modes = [1,2,3,4]
	conf.bringup_device(modes=modes)
	import mp
	import zurichHelper
	reload(zurichHelper)
