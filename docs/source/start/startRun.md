# Getting started



We assumed that you have mapped your working path into a new virtual disk M. You can start scalabrad by the bash

```bash
M:\scalabrad-0.8.3\bin\labrad --registry file:///M:/Registry?format=delphi
```

You can save the commands into a file `.bat`, and also change the file name 'scalabrad-0.8.3' and the path `M:/Registry`  (for storing experimental parameters) into yours.  Therefore, you only need to run the `.bat` file. 

The password can be assigned via `--password yourPassword` , and you can add it into the environmental variable `LABRADPASSWORD`. 



Then, start data vault `py3_data_vault.py` via bash

```bash
ipython3 py3_data_vault.py
```

The default host is `localhost`, port is `7682`. 



The other servers including `Grapher.exe`, `RegistryEditor.exe` in  `zi_labrad\labrad_server\` can be opened after you start scalabrad and data vault (Grapher require it to get data). 



You can save all the bash commands into a file, for example, `start_daily.py`

```python
import os
import win32api
import time


os.system(r'start M:\scalabrad-0.8.3\bin\labrad --registry file:///M:/Registry?format=delphi --password foo')

time.sleep(2)

os.system(r'start ipython3 py3_data_vault.py')
```

