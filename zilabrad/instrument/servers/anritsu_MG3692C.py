"""
Server for ANRITSU,MG3692C
"""
import pyvisa as visa
from functools import wraps
import logging

max_refresh = 3


def get_visa_resources():
    rm = visa.ResourceManager()
    return rm.list_resources()


class AnritsuServer(object):
    name = 'Anritsu Server'
    deviceName = ["ANRITSU MG3692C"]

    def __init__(self, address_list: list):
        self.devices = {}
        rm = visa.ResourceManager()
        for address in address_list:
            print("connecting ", str(address))
            self.devices[address] = rm.open_resource(address)
        self.rm = rm
        self.selectedDevice = self.devices.get(address)

    def add_device(self, address):
        self.devices[address] = rm.open_resource(address)

    def select_device(self, address):
        self.selectedDevice = self.devices[address]

    def refresh_device(self, dev):
        address = dev._resource_name
        rm = visa.ResourceManager()
        self.devices[address] = rm.open_resource(address)
        self.select_device(address)

    def refresh_when_error(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            dev = self.selectedDevice
            for i in range(max_refresh):
                try:
                    return func(self, dev, *args, **kwargs)
                except Exception as e:
                    if i >= max_refresh-1:
                        raise e
                    logging.warning(e)
                    self.refresh_device(self, dev)
            # return once more for safety
            return func(self, dev, *args, **kwargs)
        return wrapper

    @refresh_when_error
    def frequency(self, dev, freq=None):
        """Get or set the frequency (MHz)."""
        if freq is None:
            res = dev.query(':sour:freq?')
            return eval(res)/1e6
        else:
            dev.write(':sour:freq %f MHz' % freq)

    @refresh_when_error
    def amplitude(self, dev, amp=None):
        """Get or set the amplitude (dBm)."""
        if amp is None:
            res = dev.query(':sour:pow?')
            return eval(res)
        else:
            dev.write(':sour:pow %f' % amp)

    @refresh_when_error
    def output(self, dev, state: (None or bool) = None):
        """Get or set the output status."""
        if state is None:
            res = dev.query('outp?')
            return bool(eval(res))
        else:
            dev.write('outp %d' % int(state))
