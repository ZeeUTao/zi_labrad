# -*- coding: utf-8 -*-
"""Batched commands for the daily running.

We always open an 'ipython', and type 'run BatchRun' to start;
Then use the functions 'mp.xxx' (for example 'mp.s21_scan') to implement different experiments.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from importlib import reload
import configparser

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
def update_session(user='hwh'):
	cxn=labrad.connect()
	ss = switchSession(cxn,user=user,session=None)

	curpath = os.path.dirname(os.path.realpath(__file__))
	cfgpath = os.path.join(curpath, "zi_config.ini")
	 
	conf = configparser.ConfigParser()
	conf.read(cfgpath, encoding="utf-8")
	 
	conf.set("config", "_default_session", str(ss._dir)) 
	with open(cfgpath, "r+", encoding="utf-8") as file:
		conf.write(file)
	return ss


if __name__=="__main__":
	user = input("Enter user (default: hwh)") or "hwh"
	ss = update_session(user=user)

	do_bringup = input("Skip Bringup? (enter 0 for skip, default 1)") or 1
	if do_bringup != '0':
		_default_modes = [1,2,3,4]
		modes = _default_modes
		conf.bringup_device(modes=modes)
	import mp
	import zurichHelper
	reload(zurichHelper)


"""
if you change the session but do not want to 
BatchRun.update_session
"""