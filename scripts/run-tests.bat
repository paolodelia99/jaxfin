@ECHO OFF

call %~dp0configure-python.bat

pytest --no-header -vv --html=test_report.html --self-contained-html

pause
