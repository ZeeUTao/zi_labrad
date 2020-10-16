import os
import win32api
import time


os.system(r'start M:\scalabrad-0.8.3\bin\labrad --registry file:///M:/Registry?format=delphi --password foo')

time.sleep(2)

os.system(r'start ipython3 py3_data_vault.py')
