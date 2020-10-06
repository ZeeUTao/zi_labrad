import labrad
from pyle.registry import RegistryWrapper
from pyle.workflow import switchSession
from pyle.util import sweeptools as st

qa,hd,mw,mw_r = [None]*4


def loadInfo(paths=['Servers','devices']):
    cxn=labrad.connect()
    reg = RegistryWrapper(cxn, ['']+paths)
    return reg.copy()

def bringup_device(modes=[1,2,3,4]):
    from zurichHelper import zurich_qa,zurich_hd,microwave_source
    global qa,hd,mw,mw_r
    dev = loadInfo(paths=['Servers','devices']) ## only read
    for m in modes:
        if m == 1:
            qa = zurich_qa(dev.zi_qa_id)
        if m == 2:
            hd = zurich_hd(dev.zi_hd_id)
        if m == 3:
            mw = microwave_source(dev.microwave_source_xy)
        if m == 4:
            mw_r = microwave_source(dev.microwave_source_readout)



