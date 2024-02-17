@ECHO OFF

call %~dp0configure-python.bat

pylint jaxfin --output-format=text:pylint_res.txt,colorized

pause
