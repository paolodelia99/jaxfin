@ECHO OFF

call %~dp0configure-python.bat

pytest --cov=jaxfin tests/ --cov-report=html

pause
