<#
.SYNOPSIS
Bootstraps a Windows federated worker host for this repository.

.DESCRIPTION
Supports two Windows worker paths:

1. Docker worker onboarding for the containerized worker runtime.
2. Native worker onboarding for the DFS-lite runtime over Tailscale and SSH.

The script is idempotent. It can install Tailscale, wait for tailnet login,
enable OpenSSH Server, place an SSH public key in the correct Windows
authorized-keys file, create inbound firewall rules, optionally switch the
active network profile to Private, launch the worker, and register the worker
with the master control plane.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\scripts\windows\onboard_worker.ps1 `
  -WorkerMode Native `
  -RepoRoot (Get-Location).Path `
  -UseTailscale `
  -InstallTailscale `
  -EnsureSsh `
  -AuthorizedPublicKey "ssh-ed25519 AAAA... operator@host" `
  -SetActiveNetworkPrivate `
  -MasterEndpoint http://100.113.153.107:18080 `
  -WorkerId win_worker_1 `
  -HostPort 5000 `
  -AllowUnsupportedPython

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\scripts\windows\onboard_worker.ps1 `
  -WorkerMode Docker `
  -BuildContext .\worker `
  -BuildDockerfile .\worker\Dockerfile_extended `
  -Image hetero-fedlearn-worker-dfs:test `
  -ContainerName hetero-fedlearn-worker-1 `
  -WorkerId worker_1 `
  -HostPort 5000 `
  -SetActiveNetworkPrivate
#>
[CmdletBinding()]
param(
    [Parameter()]
    [ValidateSet("Docker", "Native")]
    [string]$WorkerMode = "Docker",

    [Parameter()]
    [string]$Image = "hetero-fedlearn-worker-dfs:test",

    [Parameter()]
    [string]$ContainerName = "hetero-fedlearn-worker",

    [Parameter()]
    [string]$WorkerId = $env:COMPUTERNAME,

    [Parameter()]
    [int]$HostPort = 5000,

    [Parameter()]
    [string]$RuleName,

    [Parameter()]
    [string]$BuildContext,

    [Parameter()]
    [string]$BuildDockerfile,

    [Parameter()]
    [string]$RepoRoot,

    [Parameter()]
    [string]$PythonBin = "py",

    [Parameter()]
    [string]$StorageDir,

    [Parameter()]
    [string]$MasterEndpoint,

    [Parameter()]
    [string]$AdvertisedEndpoint,

    [Parameter()]
    [string]$AuthorizedPublicKey,

    [Parameter()]
    [string]$AuthorizedPublicKeyPath,

    [Parameter()]
    [string]$TailscaleAuthKey,

    [Parameter()]
    [int]$TailscaleLoginTimeoutSeconds = 300,

    [switch]$PullImage,
    [switch]$SetActiveNetworkPrivate,
    [switch]$SkipDockerRun,
    [switch]$SkipWorkerLaunch,
    [switch]$SkipWorkerRegistration,
    [switch]$UseTailscale,
    [switch]$InstallTailscale,
    [switch]$EnsureSsh,
    [switch]$AllowUnsupportedPython
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $RuleName) {
    $RuleName = "HeteroFedLearn Worker Port $HostPort"
}

if ($AuthorizedPublicKey -or $AuthorizedPublicKeyPath) {
    $EnsureSsh = $true
}

<#
.SYNOPSIS
Tests whether the script is running with local administrator rights.
#>
function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

<#
.SYNOPSIS
Ensures that an external command exists on the current machine.
#>
function Assert-CommandAvailable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CommandName
    )

    if (-not (Get-Command $CommandName -ErrorAction SilentlyContinue)) {
        throw "Required command '$CommandName' is not available on PATH."
    }
}

<#
.SYNOPSIS
Returns the resolved repository root path.
#>
function Resolve-RepoRoot {
    if ($RepoRoot) {
        return (Resolve-Path $RepoRoot).Path
    }

    return (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

<#
.SYNOPSIS
Invokes Docker with strict error handling.
#>
function Invoke-Docker {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $output = & docker @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Docker command failed: docker $($Arguments -join ' ')`n$output"
    }
    return $output
}

<#
.SYNOPSIS
Invokes Tailscale with strict error handling.
#>
function Invoke-Tailscale {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [switch]$AllowFailure
    )

    $command = Get-Command tailscale -ErrorAction SilentlyContinue
    if (-not $command) {
        $candidate = Join-Path ${env:ProgramFiles} "Tailscale\tailscale.exe"
        if (Test-Path $candidate) {
            $command = Get-Item $candidate
        }
    }

    if (-not $command) {
        throw "Tailscale is not installed."
    }

    $output = & $command.Source @Arguments 2>&1
    if (-not $AllowFailure -and $LASTEXITCODE -ne 0) {
        throw "Tailscale command failed: tailscale $($Arguments -join ' ')`n$output"
    }

    return $output
}

<#
.SYNOPSIS
Creates or refreshes the inbound firewall rule for the given TCP port.
#>
function Ensure-FirewallRule {
    param(
        [Parameter(Mandatory = $true)]
        [string]$DisplayName,

        [Parameter(Mandatory = $true)]
        [int]$Port
    )

    $existingRules = @(Get-NetFirewallRule -DisplayName $DisplayName -ErrorAction SilentlyContinue)
    $matchingRule = $null
    foreach ($rule in $existingRules) {
        $portFilter = Get-NetFirewallPortFilter -AssociatedNetFirewallRule $rule | Select-Object -First 1
        if ($portFilter -and [int]$portFilter.LocalPort -eq $Port) {
            $matchingRule = $rule
            break
        }
    }

    if ($matchingRule) {
        Set-NetFirewallRule -DisplayName $DisplayName -Enabled True -Action Allow -Direction Inbound -Profile Any | Out-Null
        return
    }

    if ($existingRules.Count -gt 0) {
        $existingRules | Remove-NetFirewallRule | Out-Null
    }

    New-NetFirewallRule `
        -DisplayName $DisplayName `
        -Direction Inbound `
        -Action Allow `
        -Enabled True `
        -Protocol TCP `
        -LocalPort $Port `
        -Profile Any | Out-Null
}

<#
.SYNOPSIS
Enables the built-in Windows OpenSSH firewall rule by rule name.
#>
function Ensure-SshFirewallRule {
    $existingRule = Get-NetFirewallRule -Name "sshd" -ErrorAction SilentlyContinue
    if ($existingRule) {
        Set-NetFirewallRule -Name "sshd" -Enabled True -Action Allow -Direction Inbound -Profile Any | Out-Null
        return
    }

    New-NetFirewallRule `
        -Name "sshd" `
        -DisplayName "OpenSSH Server (sshd)" `
        -Direction Inbound `
        -Action Allow `
        -Enabled True `
        -Protocol TCP `
        -LocalPort 22 `
        -Profile Any | Out-Null
}

<#
.SYNOPSIS
Optionally switches currently connected networks to the Private category.
#>
function Set-ConnectedNetworksPrivate {
    $profiles = @(Get-NetConnectionProfile | Where-Object {
        $_.IPv4Connectivity -ne "Disconnected" -or $_.IPv6Connectivity -ne "Disconnected"
    })

    foreach ($profile in $profiles) {
        if ($profile.NetworkCategory -ne "Private") {
            Set-NetConnectionProfile -InterfaceIndex $profile.InterfaceIndex -NetworkCategory Private
        }
    }
}

<#
.SYNOPSIS
Installs Tailscale when it is missing from the host.
#>
function Ensure-TailscaleInstalled {
    if (Get-Command tailscale -ErrorAction SilentlyContinue) {
        return
    }

    if (Test-Path (Join-Path ${env:ProgramFiles} "Tailscale\tailscale.exe")) {
        return
    }

    if (Get-Command winget -ErrorAction SilentlyContinue) {
        & winget install --id Tailscale.Tailscale --silent --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) {
            throw "Tailscale installation via winget failed."
        }
        return
    }

    $installerPath = Join-Path $env:TEMP "tailscale-setup-latest-amd64.msi"
    Invoke-WebRequest -Uri "https://pkgs.tailscale.com/stable/tailscale-setup-latest-amd64.msi" -OutFile $installerPath
    $process = Start-Process -FilePath "msiexec.exe" -ArgumentList "/i", $installerPath, "/qn" -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw "Tailscale MSI installation failed with exit code $($process.ExitCode)."
    }
}

<#
.SYNOPSIS
Waits until Tailscale is connected to a tailnet.
#>
function Ensure-TailscaleConnected {
    param(
        [Parameter(Mandatory = $true)]
        [int]$TimeoutSeconds
    )

    $statusOutput = Invoke-Tailscale -Arguments @("status") -AllowFailure
    if ($LASTEXITCODE -eq 0 -and $statusOutput -notmatch "Logged out") {
        return
    }

    if ($TailscaleAuthKey) {
        Invoke-Tailscale -Arguments @("up", "--authkey=$TailscaleAuthKey")
    } else {
        $loginOutput = Invoke-Tailscale -Arguments @("up") -AllowFailure
        Write-Host $loginOutput
        Write-Host "Complete the Tailscale login in the browser, then wait for this script to continue."
    }

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 5
        $statusOutput = Invoke-Tailscale -Arguments @("status") -AllowFailure
        if ($LASTEXITCODE -eq 0 -and $statusOutput -notmatch "Logged out") {
            return
        }
    }

    throw "Tailscale did not become connected within $TimeoutSeconds seconds."
}

<#
.SYNOPSIS
Returns the first Tailscale IPv4 address for this Windows host.
#>
function Get-TailscaleIpv4 {
    $ipOutput = Invoke-Tailscale -Arguments @("ip", "-4")
    $lines = @($ipOutput | Where-Object { $_.Trim() })
    if ($lines.Count -eq 0) {
        throw "Tailscale did not return an IPv4 address."
    }
    return $lines[0].Trim()
}

<#
.SYNOPSIS
Ensures that the Windows OpenSSH Server capability is installed and enabled.
#>
function Ensure-OpenSshServer {
    $service = Get-Service -Name "sshd" -ErrorAction SilentlyContinue
    if (-not $service) {
        Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0 | Out-Null
        $service = Get-Service -Name "sshd" -ErrorAction Stop
    }

    Start-Service sshd
    Set-Service -Name sshd -StartupType Automatic
    Ensure-SshFirewallRule
}

<#
.SYNOPSIS
Returns the SSH public key content from either a direct parameter or a file.
#>
function Resolve-AuthorizedPublicKey {
    if ($AuthorizedPublicKey) {
        return $AuthorizedPublicKey.Trim()
    }

    if ($AuthorizedPublicKeyPath) {
        return (Get-Content (Resolve-Path $AuthorizedPublicKeyPath) -Raw).Trim()
    }

    return $null
}

<#
.SYNOPSIS
Determines whether the current user is in the local administrators group.
#>
function Test-CurrentUserIsAdministratorMember {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

<#
.SYNOPSIS
Installs an SSH public key into the correct Windows authorized-keys location.
#>
function Ensure-AuthorizedKey {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PublicKey
    )

    $isAdministrator = Test-CurrentUserIsAdministratorMember
    if ($isAdministrator) {
        $authorizedKeysPath = Join-Path $env:ProgramData "ssh\administrators_authorized_keys"
        if (-not (Test-Path $authorizedKeysPath)) {
            New-Item -ItemType File -Path $authorizedKeysPath -Force | Out-Null
        }

        $existingKeys = @()
        if (Test-Path $authorizedKeysPath) {
            $existingKeys = @(Get-Content $authorizedKeysPath)
        }
        if ($existingKeys -notcontains $PublicKey) {
            Add-Content -Path $authorizedKeysPath -Value $PublicKey
        }

        cmd /c icacls "%PROGRAMDATA%\ssh\administrators_authorized_keys" /inheritance:r | Out-Null
        cmd /c icacls "%PROGRAMDATA%\ssh\administrators_authorized_keys" /grant "Administrators:F" | Out-Null
        cmd /c icacls "%PROGRAMDATA%\ssh\administrators_authorized_keys" /grant "SYSTEM:F" | Out-Null
        return $authorizedKeysPath
    }

    $sshDirectory = Join-Path $HOME ".ssh"
    $authorizedKeysPath = Join-Path $sshDirectory "authorized_keys"
    New-Item -ItemType Directory -Path $sshDirectory -Force | Out-Null
    if (-not (Test-Path $authorizedKeysPath)) {
        New-Item -ItemType File -Path $authorizedKeysPath -Force | Out-Null
    }

    $existingKeys = @(Get-Content $authorizedKeysPath)
    if ($existingKeys -notcontains $PublicKey) {
        Add-Content -Path $authorizedKeysPath -Value $PublicKey
    }

    cmd /c icacls "%USERPROFILE%\.ssh" /inheritance:r | Out-Null
    cmd /c icacls "%USERPROFILE%\.ssh" /grant "%USERNAME%:(OI)(CI)F" | Out-Null
    cmd /c icacls "%USERPROFILE%\.ssh" /grant "Administrators:(OI)(CI)F" | Out-Null
    cmd /c icacls "%USERPROFILE%\.ssh\authorized_keys" /inheritance:r | Out-Null
    cmd /c icacls "%USERPROFILE%\.ssh\authorized_keys" /grant "%USERNAME%:F" | Out-Null
    cmd /c icacls "%USERPROFILE%\.ssh\authorized_keys" /grant "Administrators:F" | Out-Null

    return $authorizedKeysPath
}

<#
.SYNOPSIS
Waits until the local worker health endpoint responds successfully.
#>
function Wait-WorkerHealth {
    param(
        [Parameter(Mandatory = $true)]
        [int]$Port
    )

    $healthUri = "http://127.0.0.1:$Port/health"
    for ($attempt = 1; $attempt -le 30; $attempt++) {
        try {
            $response = Invoke-RestMethod -Method Get -Uri $healthUri -TimeoutSec 5
            if ($response.status -eq "ok") {
                return $response
            }
        } catch {
            Start-Sleep -Seconds 2
            continue
        }

        Start-Sleep -Seconds 2
    }

    throw "Worker health endpoint did not become ready at $healthUri."
}

<#
.SYNOPSIS
Registers the local worker service with the remote master control plane.
#>
function Register-WorkerWithMaster {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ResolvedMasterEndpoint,

        [Parameter(Mandatory = $true)]
        [string]$ResolvedAdvertisedEndpoint,

        [Parameter(Mandatory = $true)]
        [int]$Port
    )

    $payload = @{
        master_endpoint = $ResolvedMasterEndpoint
        advertised_endpoint = $ResolvedAdvertisedEndpoint
    } | ConvertTo-Json -Compress

    return Invoke-RestMethod `
        -Method Post `
        -Uri "http://127.0.0.1:$Port/api/connect_master" `
        -ContentType "application/json" `
        -Body $payload `
        -TimeoutSec 15
}

<#
.SYNOPSIS
Launches the DFS-lite worker natively from the checked-out repository.
#>
function Start-NativeWorker {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ResolvedRepoRoot
    )

    Assert-CommandAvailable -CommandName $PythonBin
    $storageRoot = if ($StorageDir) {
        $StorageDir
    } else {
        Join-Path $ResolvedRepoRoot "storage\$WorkerId"
    }

    $workerLogPrefix = Join-Path $ResolvedRepoRoot "worker_native_$HostPort"
    $stdoutPath = "$workerLogPrefix.out.log"
    $stderrPath = "$workerLogPrefix.err.log"
    $launcherScriptPath = "$workerLogPrefix.launch.ps1"
    $taskName = "HeteroFedLearnWorker-$HostPort"

    if (-not (Get-NetTCPConnection -LocalPort $HostPort -ErrorAction SilentlyContinue)) {
        $launcherLines = @(
            "Set-Location '$ResolvedRepoRoot'",
            "& $PythonBin start_worker.py --mode native --worker-id '$WorkerId' --host 0.0.0.0 --port $HostPort --storage-dir '$storageRoot' --no-open-browser"
        )
        if ($AllowUnsupportedPython) {
            $launcherLines[1] += " --allow-unsupported-python"
        }
        $launcherLines[1] += " 1>> '$stdoutPath' 2>> '$stderrPath'"

        Set-Content -Path $launcherScriptPath -Encoding ascii -Value ($launcherLines -join "`r`n")
        $existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
        if ($existingTask) {
            Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        }

        $taskAction = New-ScheduledTaskAction `
            -Execute "powershell.exe" `
            -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$launcherScriptPath`""
        $taskTrigger = New-ScheduledTaskTrigger -Once -At ([datetime]::Today)
        $taskSettings = New-ScheduledTaskSettingsSet `
            -AllowStartIfOnBatteries `
            -DontStopIfGoingOnBatteries `
            -StartWhenAvailable
        $currentUserName = [Security.Principal.WindowsIdentity]::GetCurrent().Name
        $taskPrincipal = New-ScheduledTaskPrincipal `
            -UserId $currentUserName `
            -LogonType Interactive `
            -RunLevel Highest

        Register-ScheduledTask `
            -TaskName $taskName `
            -Action $taskAction `
            -Trigger $taskTrigger `
            -Settings $taskSettings `
            -Principal $taskPrincipal | Out-Null
        Start-ScheduledTask -TaskName $taskName
    }

    return @{
        LauncherScriptPath = $launcherScriptPath
        TaskName = $taskName
        StorageDir = $storageRoot
        StdoutPath = $stdoutPath
        StderrPath = $stderrPath
    }
}

<#
.SYNOPSIS
Launches the DFS-lite worker in Docker and waits for readiness.
#>
function Start-DockerWorker {
    Assert-CommandAvailable -CommandName "docker"
    Invoke-Docker -Arguments @("version") | Out-Null

    if ($BuildContext) {
        if ($BuildDockerfile) {
            Invoke-Docker -Arguments @("build", "-t", $Image, "-f", $BuildDockerfile, $BuildContext) | Out-Null
        } else {
            Invoke-Docker -Arguments @("build", "-t", $Image, $BuildContext) | Out-Null
        }
    } elseif ($PullImage) {
        Invoke-Docker -Arguments @("pull", $Image) | Out-Null
    }

    if ($SkipDockerRun) {
        return
    }

    $existingContainerIds = @(Invoke-Docker -Arguments @("ps", "-aq", "--filter", "name=^$ContainerName$"))
    if ($existingContainerIds.Count -gt 0 -and ($existingContainerIds -join "").Trim()) {
        Invoke-Docker -Arguments @("rm", "-f", $ContainerName) | Out-Null
    }

    $containerArgs = @(
        "run",
        "-d",
        "--restart",
        "unless-stopped",
        "--name",
        $ContainerName,
        "-e",
        "WORKER_ID=$WorkerId",
        "-p",
        "${HostPort}:5000"
    )

    if ($StorageDir) {
        if (-not (Test-Path $StorageDir)) {
            New-Item -ItemType Directory -Path $StorageDir -Force | Out-Null
        }
        $resolvedStorageDir = (Resolve-Path $StorageDir).Path
        $containerArgs += @("-v", "${resolvedStorageDir}:/app/datanode_storage")
    }

    $containerArgs += $Image
    $containerId = (Invoke-Docker -Arguments $containerArgs | Out-String).Trim()
    return $containerId
}

if (-not (Test-IsAdministrator)) {
    throw "Run this script from an elevated PowerShell session so firewall, SSH, and network settings can be updated."
}

$resolvedRepoRoot = Resolve-RepoRoot

if ($SetActiveNetworkPrivate) {
    Set-ConnectedNetworksPrivate
}

Ensure-FirewallRule -DisplayName $RuleName -Port $HostPort

$tailscaleIpv4 = $null
if ($InstallTailscale -or $UseTailscale) {
    Ensure-TailscaleInstalled
}
if ($UseTailscale) {
    Ensure-TailscaleConnected -TimeoutSeconds $TailscaleLoginTimeoutSeconds
    $tailscaleIpv4 = Get-TailscaleIpv4
}

$authorizedKeysPath = $null
if ($EnsureSsh) {
    Ensure-OpenSshServer
    $resolvedPublicKey = Resolve-AuthorizedPublicKey
    if ($resolvedPublicKey) {
        $authorizedKeysPath = Ensure-AuthorizedKey -PublicKey $resolvedPublicKey
        Restart-Service sshd
    }
}

$resolvedAdvertisedEndpoint = $AdvertisedEndpoint
if (-not $resolvedAdvertisedEndpoint -and $UseTailscale -and $tailscaleIpv4) {
    $resolvedAdvertisedEndpoint = "http://${tailscaleIpv4}:$HostPort"
}

$workerLaunchDetails = $null
$dockerContainerId = $null
if (-not $SkipWorkerLaunch) {
    if ($WorkerMode -eq "Native") {
        $workerLaunchDetails = Start-NativeWorker -ResolvedRepoRoot $resolvedRepoRoot
    } else {
        $dockerContainerId = Start-DockerWorker
    }
}

$healthPayload = Wait-WorkerHealth -Port $HostPort

$registrationPayload = $null
if ($MasterEndpoint -and -not $SkipWorkerRegistration) {
    if (-not $resolvedAdvertisedEndpoint) {
        throw "AdvertisedEndpoint is required for registration when UseTailscale is disabled."
    }

    $registrationPayload = Register-WorkerWithMaster `
        -ResolvedMasterEndpoint $MasterEndpoint.Trim().TrimEnd("/") `
        -ResolvedAdvertisedEndpoint $resolvedAdvertisedEndpoint.Trim().TrimEnd("/") `
        -Port $HostPort
}

Write-Host "Worker onboarding completed successfully."
Write-Host "Worker Mode: $WorkerMode"
Write-Host "Worker ID: $WorkerId"
Write-Host "Health Payload: $($healthPayload | ConvertTo-Json -Compress)"
if ($tailscaleIpv4) {
    Write-Host "Tailscale IPv4: $tailscaleIpv4"
}
if ($resolvedAdvertisedEndpoint) {
    Write-Host "Advertised Endpoint: $resolvedAdvertisedEndpoint"
}
if ($MasterEndpoint) {
    Write-Host "Master Endpoint: $MasterEndpoint"
}
if ($registrationPayload) {
    Write-Host "Registration Payload: $($registrationPayload | ConvertTo-Json -Compress)"
}
if ($authorizedKeysPath) {
    Write-Host "Authorized Keys Path: $authorizedKeysPath"
}
if ($dockerContainerId) {
    Write-Host "Container ID: $dockerContainerId"
}
if ($workerLaunchDetails) {
    Write-Host "Worker task: $($workerLaunchDetails.TaskName)"
    Write-Host "Worker launch script: $($workerLaunchDetails.LauncherScriptPath)"
    Write-Host "Storage Directory: $($workerLaunchDetails.StorageDir)"
    Write-Host "Worker stdout log: $($workerLaunchDetails.StdoutPath)"
    Write-Host "Worker stderr log: $($workerLaunchDetails.StderrPath)"
}
