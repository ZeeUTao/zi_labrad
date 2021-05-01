"""
Server Driver for ANRITSU,MG3692C
"""
import pyvisa
from functools import wraps
import logging
import gc
from zilabrad.pyle.tools import Unit2SI
from zilabrad.instrument.QubitContext import qubitContext

max_refresh = 3


class AnritsuServer(object):
    """ use pyvisa to connect with device
    """

    def __init__(self, obj_name: str, address: str):
        gc.collect()
        self.deviceName = "ANRITSU MG3692C"
        self.name = obj_name
        self.address = address
        self.rm = qubitContext().get_server('pyvisa').rm
        self._connect()

    def _connect(self):
        self.device = self.rm.open_resource(self.address)

    def _refresh_device(self):
        # clear storage
        del self.device
        gc.collect()
        # create new object
        self.device = self.rm.open_resource(self.address)

    def refresh_when_error(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            for i in range(max_refresh):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    if i >= max_refresh-1:
                        raise e
                    logging.warning(e)
                    self._refresh_device()
            # return once more for safety
            return func(self, *args, **kwargs)
        return wrapper

    @refresh_when_error
    def frequency(self, freq=None):
        """Get or set the frequency (Hz)."""
        if freq is None:
            res = self.device.query(':sour:freq?')
            return eval(res)
        else:
            self.device.write(':sour:freq %f Hz' % Unit2SI(freq))

    @refresh_when_error
    def power(self, power=None):
        """Get or set the amplitude (dBm)."""
        if power is None:
            res = self.device.query(':sour:pow?')
            return eval(res)
        else:
            self.device.write(':sour:pow %f' % Unit2SI(power))

    @refresh_when_error
    def output(self, state: (None or bool) = None):
        """Get or set the output status."""
        if state is None:
            res = self.device.query('outp?')
            return bool(eval(res))
        else:
            self.device.write('outp %d' % int(state))
