@ECHO OFF

call %~dp0basedir.bat

IF EXIST "%BASEDIR%\venv" (
    REM Activate the virtual environment
    goto activation
) ELSE (
    REM Create a virtual environment
    python -m venv venv
    python -m pip install -r requirements.txt
    goto activation
)

:activation
call %BASEDIR%/venv/Scripts/activate.bat

REM black src

pylint src --output-format=text:pylint_res.txt,colorized --generate-toml-config
