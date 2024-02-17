@ECHO OFF

SET DEV=%1

call %~dp0basedir.bat

call %BASEDIR%/venv/Scripts/activate.bat

IF "%DEV%" == "" (
    SET DEV=false
) 

python -m setuptools_scm > version.txt

SET /P VERSION=<version.txt

rem Splitting the string by "."
for /f "tokens=1-3 delims=." %%a in ("%VERSION%") do (
    set "MAJOR=%%a"
    set "MINOR=%%b"
    set "PATCH=%%c"
)

ECHO "%MAJOR%.%MINOR%.%PATCH%"

IF "%DEV%"=="true" (
    SET "PROD_VERSION=%MAJOR%.%MINOR%.%PATCH%.dev"
) ELSE (
    SET "PROD_VERSION=%MAJOR%.%MINOR%.%PATCH%"
)

SET "VTAG=v%PROD_VERSION%"

ECHO "Version to be released: %PROD_VERSION%"
ECHO "Dev: %DEV%"

git tag -a %VTAG% -m "Release %PROD_VERSION%"

git push origin %VTAG%

pause