"""Batched commands for the daily running.
"""
import os
import sys

# 1. Change the path if you need
repository_zilabrad = r'M:\zi_labrad'

sys.path.append(repository_zilabrad)

from importlib import reload
import numpy as np

from zilabrad import *
from labrad.units import Unit, Value


_unitSpace = ('V', 'mV', 'us', 'ns', 's', 'GHz',
              'MHz', 'kHz', 'Hz', 'dBm', 'rad', 'None')
V, mV, us, ns, s, GHz, MHz, kHz, Hz, dBm, rad, _l = [
    Unit(s) for s in _unitSpace]
# some useful units

mp = multiplex
dp = dataProcess
# add some abbreviated to use

dh = dp.datahelp()


if __name__ == '__main__':
    from zilabrad.tests import default_parameter
    default_parameter.main()
    connect_ZI(reset=False, user='user_example')

# if __name__ == '__main__':
    # ss = connect_ZI(reset=False, user='hwh')
