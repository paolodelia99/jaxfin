@ECHO OFF

call %~dp0configure-python.bat

REM black src --check

pylint jaxfin --output-format=text:pylint_res.txt,colorized

pause
