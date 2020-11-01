@echo off
M:
cd M:\labrad_server


:loopStart
set choose=""

cls

echo This is server_start bash by Ziyu Tao
echo=
echo Enter to start 

echo 1. scalabrad
echo 2. data_vault
echo 3. gpib_server
echo 4. gpib_device_manager
echo 5. anritsu
echo a. start all
echo q. quit


set /p choose= Input the index, and press "Enter"
echo You input %choose%


if "%choose%"=="1" (
	GOTO scalabrad
) ^
else if "%choose%"=="2" (
	GOTO data_vault
) ^
else if "%choose%"=="3" (
	GOTO gpib_server
) ^
else if "%choose%"=="4" (
	GOTO gpib_device_manager
) ^
else if "%choose%"=="5" (
	GOTO anritsu
) ^
else if "%choose%"=="a" (
	GOTO startAll
) ^
else if "%choose%"=="q" (
	GOTO end
) ^
else (
    echo input invalid
)

pause
GOTO loopStart

	


:: Storing scripts
:: open scalabrad
:scalabrad
start cmd /k M:\scalabrad-0.8.3\bin\labrad --registry file:///M:/Registry?format=delphi
GOTO loopStart

:: open servers

:data_vault
start cmd /k ipython3 py3_data_vault.py
GOTO loopStart

:gpib_server
start cmd /k ipython3 gpib_server.py
GOTO loopStart

:gpib_device_manager
start cmd /k ipython3 gpib_device_manager.py
GOTO loopStart

:anritsu
start cmd /k ipython3 anritsu.py
GOTO loopStart

:startAll
:: open scalabrad
start cmd /k M:\scalabrad-0.8.3\bin\labrad --registry file:///M:/Registry?format=delphi
TIMEOUT /T 5
start cmd /k ipython3 py3_data_vault.py
start cmd /k ipython3 gpib_server.py
start cmd /k ipython3 gpib_device_manager.py
start cmd /k ipython3 anritsu.py
GOTO loopStart


:end