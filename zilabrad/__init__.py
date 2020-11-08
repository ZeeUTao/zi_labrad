

"""Set up the main namespace."""

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


# -----------------------------------------------------------------------------
# Load modules
#
from zilabrad.instrument import waveforms

# example: ar[0:2:0.1,GHz]
from zilabrad.pyle.util import sweeptools
ar = sweeptools.RangeCreator()

from zilabrad.instrument import zurichHelper
from zilabrad.instrument import qubitServer
from zilabrad.instrument.QubitContext import qubitContext
from zilabrad.instrument.QubitContext import update_session
from zilabrad.plots import dataProcess
from zilabrad import multiplex


def connect_ZI():
    global session
    user = input("Enter user (default: hwh)") or "hwh"
    session = update_session(user=user)
    do_bringup = input("Skip Bringup? (enter 0 for skip, default 1)") or 1
    if do_bringup != '0':
        qctx = qubitContext()
        qctx.refresh()
    return


# -----------------------------------------------------------------------------
# Clean name space
#

