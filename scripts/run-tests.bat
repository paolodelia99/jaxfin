@ECHO OFF

call %~dp0configure-python.bat

pytest --junit-xml=test_report.xml

pause
