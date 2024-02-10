@ECHO OFF

call %~dp0configure-python.bat

black jaxfin --check

pause