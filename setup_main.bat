@echo off
ECHO If you have not yet installed Python install manager please close this window and do so.
pause
:BEG
IF NOT EXIST "%~dp0translator\" (
	ECHO Setting up Python 3.11 enviroment
	py -3.11 -m venv translator
) ELSE (
	ECHO Enviroment already seems to exist
	choice /C YN /M "Would you like to create a new clean enviroment? (WARNING: This will delete the existing one)"
	if errorlevel 2 goto :NO
	rmdir /S /Q /F "%~dp0translator\"
	goto :BEG
)
:NO
ECHO Activating enviroment
call translator/Scripts/activate.bat
ECHO Installing prerequisites
python pip install -r requirements.txt