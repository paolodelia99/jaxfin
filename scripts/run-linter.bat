@ECHO OFF

call %~dp0configure-python.bat

ruff check jaxfin --output-format=full

pause
