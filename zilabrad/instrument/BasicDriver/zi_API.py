
import zhinst.utils  # create API object
from zilabrad.pyle.tools import singleton

@singleton
class ziDAQ(object):
    """singleton class for zurich daq
    """
    def __init__(self,obj_name='ziDAQ',labone_ip='localhost'):
        # connectivity must 8004 for zurish instruments
        self.labone_ip = labone_ip
        self.refresh_api()
        self.secret_mode(mode=1) ## (default) daq can be used by everyone.

    def secret_mode(self, mode=1):
        # mode = 1: daq can be created by everyone
        # mode = 0: daq only be used by localhost
        self.daq.setInt('/zi/config/open', mode)

    def refresh_api(self,labone_ip=None):
        if not isinstance(labone_ip,type(None)):
            self.labone_ip = labone_ip
        self.daq = zhinst.ziPython.ziDAQServer(self.labone_ip,8004,6)

