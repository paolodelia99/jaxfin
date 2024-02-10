@ECHO OFF

call %~dp0configure-python.bat

mypy jaxfin

pause
