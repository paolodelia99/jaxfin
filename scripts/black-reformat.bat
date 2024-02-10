@ECHO OFF

call %~dp0configure-python.bat

black src

pause