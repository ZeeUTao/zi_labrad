"""
basic Driver for communication of NI-visa instrument
"""
import pyvisa
from zilabrad.pyle.tools import singleton



@singleton
class visa(object):
    """ singleton class for pyvisa.ResourceManager()
    """
    def __init__(self,obj_name='pyvisa',address=None):
        self.rm = pyvisa.ResourceManager()

    def get_visa_resources(self):
        return self.rm.list_resources()
