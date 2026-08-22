<#
.SYNOPSIS
    Launches the VideoCraft AI WebUI using Streamlit in PowerShell.

.DESCRIPTION
    VideoCraft AI WebUI Launcher for Windows PowerShell.
    Automatically detects Python virtual environment (.venv), uv, or system Streamlit,
    resolves an available TCP port (8501-8599), and starts the WebUI server.

.PARAMETER HostAddress
    The IP / Hostname to bind the WebUI server (default: 127.0.0.1 or $env:MPT_WEBUI_HOST).

.PARAMETER Port
    The preferred TCP port to use (default: 8501 or $env:MPT_WEBUI_PORT).

.PARAMETER NoBrowser
    If set, prevents opening the default browser automatically.

.EXAMPLE
    .\webui.ps1
    .\webui.ps1 -Port 8505
    .\webui.ps1 -HostAddress 0.0.0.0 -Port 8501
#>

[CmdletBinding()]
param (
    [string]$HostAddress = $env:MPT_WEBUI_HOST,
    [int]$Port = 0,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $ScriptDir

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "             ✨ VideoCraft AI WebUI Launcher ✨             " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Working Directory: $ScriptDir" -ForegroundColor Gray

# Set PYTHONPATH to project root
$env:PYTHONPATH = $ScriptDir

# Determine Host
if ([string]::IsNullOrWhiteSpace($HostAddress)) {
    $HostAddress = "127.0.0.1"
}

# Determine Preferred Port
if ($Port -eq 0) {
    if (![string]::IsNullOrWhiteSpace($env:MPT_WEBUI_PORT) -and [int]::TryParse($env:MPT_WEBUI_PORT, [ref]$Port)) {
        # Used from env
    } else {
        $Port = 8501
    }
}

# Find Streamlit executable / command
$StreamlitCmd = $null
$PythonExe = $null

$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"
$LibPython = Join-Path $ScriptDir "lib\python\python.exe"

if (Test-Path $VenvPython) {
    $PythonExe = $VenvPython
    $StreamlitCmd = @($VenvPython, "-m", "streamlit")
    Write-Host " [Environment] Using virtualenv Python: $VenvPython" -ForegroundColor Green
} elseif (Test-Path $LibPython) {
    $PythonExe = $LibPython
    $StreamlitCmd = @($LibPython, "-m", "streamlit")
    Write-Host " [Environment] Using embedded Python: $LibPython" -ForegroundColor Green
} elseif (Get-Command uv -ErrorAction SilentlyContinue) {
    $StreamlitCmd = @("uv", "run", "streamlit")
    Write-Host " [Environment] Using 'uv run streamlit'" -ForegroundColor Green
} elseif (Get-Command streamlit -ErrorAction SilentlyContinue) {
    $StreamlitCmd = @("streamlit")
    Write-Host " [Environment] Using system 'streamlit' on PATH" -ForegroundColor Yellow
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $StreamlitCmd = @("python", "-m", "streamlit")
    Write-Host " [Environment] Using system 'python -m streamlit'" -ForegroundColor Yellow
} else {
    Write-Host " [Error] Neither Python virtualenv, uv, nor streamlit was found." -ForegroundColor Red
    Write-Host " Please install dependencies first (e.g., run 'uv sync' or 'pip install -r requirements.txt')." -ForegroundColor Yellow
    Exit 1
}

# Find available port
function Find-AvailablePort([string]$hostStr, [int]$preferredPort) {
    $hostIP = $null
    try {
        $addresses = [System.Net.Dns]::GetHostAddresses($hostStr)
        foreach ($addr in $addresses) {
            if ($addr.AddressFamily -eq [System.Net.Sockets.AddressFamily]::InterNetwork) {
                $hostIP = $addr
                break
            }
        }
    } catch {
        $hostIP = [System.Net.IPAddress]::Parse("127.0.0.1")
    }

    if ($null -eq $hostIP) {
        $hostIP = [System.Net.IPAddress]::Parse("127.0.0.1")
    }

    $candidates = [System.Collections.Generic.List[int]]::new()
    $candidates.Add($preferredPort)
    foreach ($cand in 8502..8599) {
        if ($cand -ne $preferredPort) {
            $candidates.Add($cand)
        }
    }

    foreach ($candidatePort in $candidates) {
        $socket = [System.Net.Sockets.Socket]::new(
            [System.Net.Sockets.AddressFamily]::InterNetwork,
            [System.Net.Sockets.SocketType]::Stream,
            [System.Net.Sockets.ProtocolType]::Tcp
        )
        try {
            $socket.Bind([System.Net.IPEndPoint]::new($hostIP, $candidatePort))
            $socket.Close()
            return $candidatePort
        } catch {
            try { $socket.Close() } catch {}
        }
    }
    return $null
}

$SelectedPort = Find-AvailablePort -hostStr $HostAddress -preferredPort $Port
if ($null -eq $SelectedPort) {
    Write-Host " [Error] No available WebUI port found in range 8501-8599 for $HostAddress." -ForegroundColor Red
    Exit 1
}

if ($SelectedPort -ne $Port) {
    Write-Host " [Notice] Port $Port was in use. Using port $SelectedPort instead." -ForegroundColor Yellow
}

$env:MPT_WEBUI_HOST = $HostAddress
$env:MPT_WEBUI_PORT = "$SelectedPort"

$WebUIUrl = "http://${HostAddress}:${SelectedPort}"
Write-Host " WebUI Address: $WebUIUrl" -ForegroundColor Cyan
Write-Host " Starting WebUI Server..." -ForegroundColor Green
Write-Host "----------------------------------------------------------" -ForegroundColor Gray

# Streamlit run arguments
$MainAppPath = Join-Path $ScriptDir "webui\Main.py"
$StreamlitArgs = @(
    "run",
    $MainAppPath,
    "--server.address=$HostAddress",
    "--server.port=$SelectedPort",
    "--browser.serverAddress=$HostAddress",
    "--browser.gatherUsageStats=False",
    "--client.toolbarMode=minimal",
    "--logger.hideWelcomeMessage=True",
    "--server.showEmailPrompt=False",
    "--server.enableCORS=True"
)

if ($StreamlitCmd.Count -gt 1) {
    $ExecCmd = $StreamlitCmd[0]
    $AllArgs = $StreamlitCmd[1..($StreamlitCmd.Count - 1)] + $StreamlitArgs
    & $ExecCmd $AllArgs
} else {
    $ExecCmd = $StreamlitCmd[0]
    & $ExecCmd $StreamlitArgs
}
