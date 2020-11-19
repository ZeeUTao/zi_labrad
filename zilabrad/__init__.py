"""
Modular quantum control for zurich instrument
"""


# set up the main namespace.
from zilabrad import multiplex
from zilabrad.plots import dataProcess
from zilabrad.instrument.QubitContext import update_session
from zilabrad.instrument.QubitContext import qubitContext
from zilabrad.instrument import qubitServer
from zilabrad.instrument import zurichHelper

from zilabrad.instrument import waveforms
from zilabrad.pyle.util import sweeptools
from zilabrad.util import clear_singletonMany

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
    'connect_ZI',
    # for developer
    'qubitContext',
    'qubitServer',
    'waveforms',
    'zurichHelper',
]


def connect_ZI(reset=False):
    if reset:
        clear_singletonMany(zurichHelper.zurich_qa)
        clear_singletonMany(zurichHelper.zurich_hd)

    user = input("Enter user (default: hwh)") or "hwh"
    sample = update_session(user=user)
    do_bringup = input("Skip Bringup? (enter 0 for skip, default 1)") or 1
    if do_bringup != '0':
        qctx = qubitContext()
        qctx.refresh()
    return sample
