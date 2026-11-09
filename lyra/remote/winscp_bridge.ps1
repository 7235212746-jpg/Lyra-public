$ErrorActionPreference = "Stop"

function Write-Result([hashtable]$Value) {
    [Console]::Out.Write(($Value | ConvertTo-Json -Depth 8 -Compress))
}

function Quote-Shell([string]$Value) {
    return "'" + $Value.Replace("'", "'`"'`"'") + "'"
}

function Open-LyraSession($Request) {
    $options = New-Object WinSCP.SessionOptions
    $options.Protocol = [WinSCP.Protocol]::Sftp
    $options.HostName = [string]$Request.profile.host
    $options.PortNumber = [int]$Request.profile.port
    $options.UserName = [string]$Request.profile.username
    $options.SshHostKeyFingerprint = [string]$Request.profile.host_fingerprint
    if ($Request.profile.private_key_path) {
        $options.SshPrivateKeyPath = [string]$Request.profile.private_key_path
    }
    if ($Request.authentication.password) {
        $options.Password = [string]$Request.authentication.password
    }
    if ($Request.authentication.private_key_passphrase) {
        $options.PrivateKeyPassphrase = [string]$Request.authentication.private_key_passphrase
    }
    $options.Timeout = [TimeSpan]::FromSeconds(30)
    $session = New-Object WinSCP.Session
    $session.ExecutablePath = [string]$Request.exe_path
    $session.Open($options)
    return $session
}

try {
    $requestText = [Console]::In.ReadToEnd()
    if ($requestText.Length -gt 1048576) { throw "request exceeds size limit" }
    $request = $requestText | ConvertFrom-Json
    Add-Type -Path ([string]$request.dll_path)

    if ($request.operation -eq "scan") {
        $options = New-Object WinSCP.SessionOptions
        $options.Protocol = [WinSCP.Protocol]::Sftp
        $options.HostName = [string]$request.profile.host
        $options.PortNumber = [int]$request.profile.port
        $options.Timeout = [TimeSpan]::FromSeconds(20)
        $session = New-Object WinSCP.Session
        $session.ExecutablePath = [string]$request.exe_path
        try {
            $fingerprint = $session.ScanFingerprint($options, "SHA-256")
        }
        finally { $session.Dispose() }
        Write-Result @{ ok = $true; fingerprint = $fingerprint }
        exit 0
    }

    $session = Open-LyraSession $request
    try {
        switch ([string]$request.operation) {
            "home" {
                Write-Result @{ ok = $true; home = [string]$session.HomePath }
            }
            "list" {
                $items = @()
                foreach ($item in $session.ListDirectory([string]$request.remote_path).Files) {
                    if ($item.Name -in @(".", "..")) { continue }
                    $milliseconds = [DateTimeOffset]$item.LastWriteTime
                    $items += @{
                        name = [string]$item.Name
                        path = [string]$item.FullName
                        is_directory = [bool]$item.IsDirectory
                        size = [long][Math]::Max(0, $item.Length)
                        modified_ns = [long]($milliseconds.ToUnixTimeMilliseconds() * 1000000)
                    }
                }
                Write-Result @{ ok = $true; entries = $items }
            }
            "get" {
                $info = $session.GetFileInfo([string]$request.remote_path)
                if ([long]$info.Length -gt [long]$request.max_bytes) {
                    throw "remote payload exceeds configured bound"
                }
                $result = $session.GetFiles(
                    [string]$request.remote_path,
                    [string]$request.local_path,
                    $false
                )
                $result.Check()
                Write-Result @{ ok = $true; size = [long]$info.Length }
            }
            "put-atomic" {
                $remotePath = [string]$request.remote_path
                $slash = $remotePath.LastIndexOf("/")
                $remoteDirectory = if ($slash -le 0) { "/" } else { $remotePath.Substring(0, $slash) }
                $mkdir = $session.ExecuteCommand("mkdir -p -- " + (Quote-Shell $remoteDirectory))
                if ($mkdir.ExitCode -ne 0) { throw "cannot create remote cache directory" }
                $temporaryRemote = $remotePath + ".lyra-upload-" + [Guid]::NewGuid().ToString("N")
                $transfer = New-Object WinSCP.TransferOptions
                $transfer.TransferMode = [WinSCP.TransferMode]::Binary
                $permissions = New-Object WinSCP.FilePermissions
                $permissions.Numeric = [string]$request.mode
                $transfer.FilePermissions = $permissions
                try {
                    $result = $session.PutFiles(
                        [string]$request.local_path,
                        $temporaryRemote,
                        $false,
                        $transfer
                    )
                    $result.Check()
                    $session.MoveFile($temporaryRemote, $remotePath)
                }
                catch {
                    try { $session.RemoveFile($temporaryRemote) } catch {}
                    throw
                }
                Write-Result @{ ok = $true }
            }
            "start-job" {
                $command = "nohup python3 " + (Quote-Shell ([string]$request.agent_path)) +
                    " " + (Quote-Shell ([string]$request.job_id)) +
                    " </dev/null >/dev/null 2>&1 &"
                $result = $session.ExecuteCommand($command)
                if ($result.ExitCode -ne 0) { throw "remote Lyra supervisor failed to start" }
                Write-Result @{ ok = $true }
            }
            "runtime-preflight" {
                $command = "python3 " + (Quote-Shell ([string]$request.script_path)) +
                    " " + (Quote-Shell ([string]$request.cache_root))
                $result = $session.ExecuteCommand($command)
                if ($result.ExitCode -ne 0) { throw "remote Lyra preflight failed" }
                if ($result.Output.Length -gt 65536) { throw "remote preflight report exceeds size limit" }
                Write-Result @{ ok = $true; report = [string]$result.Output }
            }
            default { throw "unsupported WinSCP bridge operation" }
        }
    }
    finally { $session.Dispose() }
}
catch {
    Write-Result @{ ok = $false; error = [string]$_.Exception.Message }
    exit 1
}
