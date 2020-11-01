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



## Bash

You can save all the bash commands into a file, for example

```bash
start cmd /k M:\scalabrad-0.8.3\bin\labrad --registry file:///M:/Registry?format=delphi
TIMEOUT /T 5
start cmd /k ipython3 py3_data_vault.py
start cmd /k ipython3 gpib_server.py
start cmd /k ipython3 gpib_device_manager.py
start cmd /k ipython3 anritsu.py
```



An example of batch file is given in `zi_labrad\labrad_server\bat-script\server_start.bat` , which provides an interface like

```bash
This is server_start bash

Enter to start
1. scalabrad
2. data_vault
3. gpib_server
4. gpib_device_manager
5. anritsu
a. start all
q. quit
Input the index, and press "Enter"
```

You can modify the batch file and customize your own commands for the daily works. 