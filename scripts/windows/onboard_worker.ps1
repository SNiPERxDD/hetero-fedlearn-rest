<#
.SYNOPSIS
Bootstraps a Windows federated worker host for this repository.

.DESCRIPTION
Creates or refreshes the required inbound firewall rule, can optionally switch
the active network profile to Private, can optionally build or pull the worker
image, and can launch the worker container with a health check confirmation.

.EXAMPLE
pwsh -ExecutionPolicy Bypass -File .\scripts\windows\onboard_worker.ps1 `
  -BuildContext .\worker `
  -Image hetero-fedlearn-worker:test `
  -ContainerName hetero-fedlearn-worker-1 `
  -WorkerId worker_1 `
  -HostPort 5000 `
  -SetActiveNetworkPrivate

.EXAMPLE
pwsh -ExecutionPolicy Bypass -File .\scripts\windows\onboard_worker.ps1 `
  -PullImage `
  -Image ghcr.io/example/hetero-fedlearn-worker:latest `
  -ContainerName hetero-fedlearn-worker-2 `
  -WorkerId worker_2 `
  -HostPort 5000
#>
[CmdletBinding()]
param(
    [Parameter()]
    [string]$Image = "hetero-fedlearn-worker:test",

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

    [switch]$PullImage,
    [switch]$SetActiveNetworkPrivate,
    [switch]$SkipDockerRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $RuleName) {
    $RuleName = "HeteroFedLearn Worker Port $HostPort"
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
Creates or refreshes the inbound firewall rule for the worker port.
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
        Set-NetFirewallRule -DisplayName $DisplayName -Enabled True -Action Allow -Direction Inbound -Profile Private,Domain | Out-Null
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
        -Profile Private,Domain | Out-Null
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
Waits until the local worker health endpoint responds successfully.
#>
function Wait-WorkerHealth {
    param(
        [Parameter(Mandatory = $true)]
        [int]$Port
    )

    $healthUri = "http://127.0.0.1:$Port/health"
    for ($attempt = 1; $attempt -le 20; $attempt++) {
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

if (-not (Test-IsAdministrator)) {
    throw "Run this script from an elevated PowerShell session so firewall and network settings can be updated."
}

Assert-CommandAvailable -CommandName "docker"
Invoke-Docker -Arguments @("version") | Out-Null

if ($SetActiveNetworkPrivate) {
    Set-ConnectedNetworksPrivate
}

Ensure-FirewallRule -DisplayName $RuleName -Port $HostPort

if ($BuildContext) {
    Invoke-Docker -Arguments @("build", "-t", $Image, $BuildContext) | Out-Null
} elseif ($PullImage) {
    Invoke-Docker -Arguments @("pull", $Image) | Out-Null
}

if (-not $SkipDockerRun) {
    $existingContainerIds = @(Invoke-Docker -Arguments @("ps", "-aq", "--filter", "name=^$ContainerName$"))
    if ($existingContainerIds.Count -gt 0 -and ($existingContainerIds -join "").Trim()) {
        Invoke-Docker -Arguments @("rm", "-f", $ContainerName) | Out-Null
    }

    $containerId = (Invoke-Docker -Arguments @(
        "run",
        "-d",
        "--restart",
        "unless-stopped",
        "--name",
        $ContainerName,
        "-e",
        "WORKER_ID=$WorkerId",
        "-p",
        "${HostPort}:5000",
        $Image
    ) | Out-String).Trim()

    $healthPayload = Wait-WorkerHealth -Port $HostPort
    Write-Host "Worker onboarding completed successfully."
    Write-Host "Container: $ContainerName"
    Write-Host "Container ID: $containerId"
    Write-Host "Worker ID: $WorkerId"
    Write-Host "Firewall Rule: $RuleName"
    Write-Host "Health Payload: $($healthPayload | ConvertTo-Json -Compress)"
} else {
    Write-Host "Firewall and prerequisite onboarding completed. Docker run step was skipped."
    Write-Host "Worker ID: $WorkerId"
    Write-Host "Firewall Rule: $RuleName"
}
[comment-based help intentionally omitted?]
