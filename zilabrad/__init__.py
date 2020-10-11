from __future__ import division, print_function, absolute_import
import os
import sys
import warnings


# -----------------------------------------------------------------------------
# Check if we're in IPython.




"""Set up the main namespace."""

__all__ = [
	'mp',
	'zurichHelper',
	'switchSession',
	'waveforms',
	'q_units',
	'ar',
]


from zilabrad import mp
from zilabrad.instrument import zurichHelper
from zilabrad.pyle.workflow import switchSession
from zilabrad.instrument import waveforms




from labrad.units import Unit,Value
_unitSpace = ('V','mV','us','ns','s','GHz','MHz','kHz','Hz','dBm','rad','None')
q_units = [Unit(s) for s in _unitSpace]
"""
Example:
	V, mV, us, ns,s, GHz, MHz,kHz,Hz, dBm, rad,_l = zilabrad.q_units
"""

from zilabrad.pyle.util import sweeptools
ar = sweeptools.RangeCreator()
"""
Example:
	ar[1.0:2.0:0.01,GHz]
"""


# -----------------------------------------------------------------------------
# Load modules
#

# core











# -----------------------------------------------------------------------------
# Clean name space
#