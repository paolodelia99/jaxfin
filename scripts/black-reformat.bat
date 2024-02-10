@ECHO OFF

call %~dp0configure-python.bat

black jaxfin

pause