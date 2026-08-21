<#
.SYNOPSIS
    Build and run the production Docker image (root Dockerfile) locally,
    as a stand-in for .github/workflows/deploy-gke.yml's build-and-push +
    deploy jobs -- minus GCP/GKE, entirely on this machine.

.DESCRIPTION
    This is NOT the same thing as `docker compose up` (docker-compose.yml):
    that runs two hot-reloading dev containers built from raw base images
    (backend on :8000 installing deps at startup, frontend vite dev server
    on :5173). This script builds the multi-stage Dockerfile instead -- one
    image with the frontend already built in and served by FastAPI on a
    single port. Defaults to host port 8081 so it can run alongside the
    docker-compose dev stack without a port clash; pass -Port to change it.

    Data (sqlite db, uploads) persists in a named Docker volume across runs.
    Use -Down to stop/remove the container (volume is kept; `docker volume
    rm claims-portal-local-data` wipes it).

.PARAMETER Port
    Host port to bind to the container's :8000. Default 8081.

.PARAMETER NoCache
    Pass --no-cache to `docker build`.

.PARAMETER Down
    Stop and remove the running container, then exit.

.PARAMETER Logs
    Tail the running container's logs, then exit.

.EXAMPLE
    ./scripts/deploy-local.ps1
.EXAMPLE
    ./scripts/deploy-local.ps1 -Port 8080 -NoCache
.EXAMPLE
    ./scripts/deploy-local.ps1 -Down
#>

param(
    [int]$Port = 8081,
    [switch]$NoCache,
    [switch]$Down,
    [switch]$Logs
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$ImageTag = "claims-portal:local"
$ContainerName = "claims-portal-local"
$VolumeName = "claims-portal-local-data"
$EnvFile = Join-Path $RepoRoot ".env"

if ($Down) {
    # try/catch, not just 2>$null: on Windows PowerShell 5.1, redirecting a
    # native command's stderr wraps it in a terminating NativeCommandError
    # under $ErrorActionPreference = "Stop" -- even for an expected "no such
    # container" message -- so the redirect alone doesn't make this safe.
    try { docker rm -f $ContainerName 2>$null | Out-Null } catch {}
    Write-Host "Stopped and removed $ContainerName. Volume '$VolumeName' kept -- run 'docker volume rm $VolumeName' to also wipe its data."
    exit 0
}

if ($Logs) {
    docker logs -f $ContainerName
    exit 0
}

# Only pass through keys the app actually reads via Settings/os.getenv
# (see backend/app/core/config.py). Deliberately NOT DATABASE_URL /
# UPLOAD_DIR / MODEL_DIR / DATA_DIR: the Dockerfile bakes those in as
# absolute container paths (/data/...) matching the volume mounted below,
# while .env's versions are relative paths meant for docker-compose.yml's
# bind-mount dev setup -- passing those through here would silently point
# the app at ephemeral, unmounted paths instead of the persisted volume.
$PassthroughKeys = @(
    "GROQ_API_KEY", "MODEL_NAME", "GROQ_BASE_URL",
    "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL",
    "JWT_SECRET_KEY", "JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "JWT_REFRESH_TOKEN_EXPIRE_DAYS"
)

$DockerRunArgs = @("run", "-d", "--name", $ContainerName, "-p", "${Port}:8000", "-v", "${VolumeName}:/data")

if (Test-Path $EnvFile) {
    $EnvValues = @{}
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
            $EnvValues[$Matches[1]] = $Matches[2].Trim().Trim('"')
        }
    }
    foreach ($key in $PassthroughKeys) {
        if ($EnvValues.ContainsKey($key) -and $EnvValues[$key]) {
            $DockerRunArgs += @("-e", "$key=$($EnvValues[$key])")
        }
    }
} else {
    Write-Warning ".env not found at $EnvFile -- running without GROQ/Langfuse keys (deterministic fallback reports, observability off)."
}

$DockerRunArgs += $ImageTag

Write-Host "Building $ImageTag from Dockerfile..."
$BuildArgs = @("build", "-t", $ImageTag)
if ($NoCache) { $BuildArgs += "--no-cache" }
$BuildArgs += $RepoRoot

& docker @BuildArgs
if ($LASTEXITCODE -ne 0) { Write-Error "docker build failed."; exit 1 }

try { docker rm -f $ContainerName 2>$null | Out-Null } catch {}

Write-Host "Starting $ContainerName -> http://localhost:$Port ..."
& docker @DockerRunArgs
if ($LASTEXITCODE -ne 0) { Write-Error "docker run failed."; exit 1 }

# Smoke test -- same idea as the 'Smoke test' step in deploy-gke.yml, just
# polling the locally-published port directly instead of a kubectl
# port-forward.
Write-Host "Waiting for /health..."
$Healthy = $false
for ($i = 0; $i -lt 20; $i++) {
    try {
        $Response = Invoke-WebRequest -Uri "http://localhost:$Port/health" -UseBasicParsing -TimeoutSec 3
        if ($Response.StatusCode -eq 200) { $Healthy = $true; break }
    } catch {}
    Start-Sleep -Seconds 3
}

if ($Healthy) {
    Write-Host "Healthy. App running at http://localhost:$Port"
} else {
    Write-Error "Smoke test failed: /health never became reachable. Check logs with: docker logs $ContainerName"
    exit 1
}
