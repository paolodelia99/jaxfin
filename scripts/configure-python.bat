ECHO OFF

call %~dp0basedir.bat

IF "%VIRTUAL_ENV%" NEQ "" goto end

IF EXIST "%BASEDIR%\venv" (
    REM Activate the virtual environment
    goto activation
) ELSE (
    REM Create a virtual environment
    python -m venv venv
    goto activation
)

:activation
call %BASEDIR%/venv/Scripts/activate.bat
python -m pip install -r requirements/build.txt

:end