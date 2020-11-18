@echo off

set /p choose= Input your IP, and press "Enter"
if "%choose%"=="" (
	echo You input nothing
) ^
else (
	echo You input %choose%
	setx LABRADhost %choose%
	echo LABRADhost %choose%
) ^


setx LABRADPASSWORD pass
echo LABRADPASSWORD %LABRADPASSWORD%

setx LABRADport 7682
echo LABRADport %LABRADport%

pause