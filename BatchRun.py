import matplotlib.pyplot as plt ## give picture
import numpy as np
import conf

from importlib import reload

import labrad
from pyle.workflow import switchSession
from pyle.util import sweeptools as st
from labrad.units import Unit,Value
_unitSpace = ('V', 'mV', 'us', 'ns','s', 'GHz', 'MHz','kHz','Hz', 'dBm', 'rad','None')
V, mV, us, ns,s, GHz, MHz,kHz,Hz, dBm, rad,_l  = [Unit(s) for s in _unitSpace]
ar = st.r


if __name__=="__main__":
	cxn=labrad.connect()
	ss = switchSession(cxn,user='hwh',session=None) 

	modes = [1,2,3,4]
	conf.bringup_device(modes=modes)
	import mp
	import zurichHelper
	reload(zurichHelper)
