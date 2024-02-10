@ECHO OFF

call %~dp0configure-python.bat

pytest --html=test_report.html --self-contained-html

pause
