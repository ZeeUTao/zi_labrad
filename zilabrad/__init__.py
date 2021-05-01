"""
Modular quantum control for zurich instrument
"""


# set up the main namespace.
import labrad
from zilabrad.instrument.QubitContext import qubitContext
from zilabrad.instrument import zurichHelper
from zilabrad.instrument import waveforms

from zilabrad.pyle.util import sweeptools
from zilabrad.pyle.workflow import switchSession


from zilabrad import multiplex
from zilabrad.plots import dataProcess

"""
Author: Ziyu Tao, WenHui Huang
Git maintainer: Ziyu Tao
"""

ar = sweeptools.RangeCreator()
# example usage: ar[0:2:0.1,GHz]

__all__ = [
    'ar',
    'multiplex',
    'dataProcess',
    'login',
    # for developer
    'qubitContext',
    'qubitServer',
    'waveforms',
    'zurichHelper',
]

cxn = labrad.connect()


def login(user='hwh', bringup=False):
    # make sure login user name
    user_input = input(f"Enter user (default: {user})") or user
    # load sample register from local
    sample = switchSession(cxn, user=user, session=None)
    # create device management instance and give sample
    qc = qubitContext(cxn=cxn, user_sample=sample)
    if bringup:
        qc.refresh()
    return sample
