@echo off
if "%1" == "h" goto begin
mshta vbscript:createobject("wscript.shell").run("""%~nx0"" h",0)(window.close)&&exit
:begin
REM

M:\scalabrad-0.8.3\bin\labrad --registry file:///M:/Registry?format=delphi