@echo off
setlocal enabledelayedexpansion

REM Check if the authentication token is provided as an argument
if "%~1"=="" (
    echo Usage: %~nx0 ^<AUTH_TOKEN^>
    exit /b 1
)

REM Loop through each id from twitch-dl output
for /f "usebackq tokens=*" %%a in (`twitch-dl videos perokichi_neet --all --json ^| jq -r ".videos[].id"`) do (
    REM Remove surrounding double quotes from the id
    set "id=%%~a"

    if not exist ".\vods\!id!.mkv" (
        REM Call twitch-dl download with the id and the provided AUTH_TOKEN
        twitch-dl download -q 160p --auth-token "%~1" --output ".\vods\!id!.{format}" !id!
    )
)