@echo off
IF NOT EXIST "%~dp0translator\" (
	ECHO ERROR: Enviroment not set up.
	Echo Please use the setup file or set up an enviroment manually
	pause
	exit
)
ECHO Activating enviroment
call translator/Scripts/activate.bat
ECHO Running the app
python translator_client.py