param([string]$OutputPath, [string]$StopPath, [switch]$Once)
$ErrorActionPreference = 'Stop'
# Only public resource counters; never inspect process environments or credentials.
$monitorStatePath = $OutputPath + '.monitor.json'
$monitorState = [ordered]@{
    pid_namespace = 'windows'
    windows_pid = $PID
    process_started_utc = (Get-Process -Id $PID).StartTime.ToUniversalTime().ToString('o')
    state = 'starting'
    sample_count = 0
    last_success_unix_seconds = $null
    exit_reason = $null
    error_type = $null
    error_hresult = $null
    error_message = $null
    error_id = $null
    error_stack = $null
    error_line = $null
    source = 'GlobalMemoryStatusEx'
    total_physical_bytes = $null
    available_physical_bytes = $null
    memory_load_percent = $null
}

function Write-MonitorState {
    $stateTemporary = $monitorStatePath + '.next'
    [IO.File]::WriteAllText($stateTemporary, ($monitorState | ConvertTo-Json -Compress), [Text.UTF8Encoding]::new($false))
    if ([IO.File]::Exists($monitorStatePath)) { [IO.File]::Replace($stateTemporary, $monitorStatePath, [System.Management.Automation.Language.NullString]::Value) }
    else { [IO.File]::Move($stateTemporary, $monitorStatePath) }
}

function Initialize-NativeMemory {
    if (-not ('Wardrowbe.MemoryNative' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
namespace Wardrowbe {
    [StructLayout(LayoutKind.Sequential)]
    public struct MemoryStatus {
        public uint Length;
        public uint MemoryLoad;
        public ulong TotalPhysical;
        public ulong AvailablePhysical;
        public ulong TotalPageFile;
        public ulong AvailablePageFile;
        public ulong TotalVirtual;
        public ulong AvailableVirtual;
        public ulong AvailableExtendedVirtual;
    }
    public static class MemoryNative {
        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool GlobalMemoryStatusEx(ref MemoryStatus status);
    }
}
'@
    }
}

function Read-NativeMemory {
    $memory = New-Object Wardrowbe.MemoryStatus
    $memory.Length = [Runtime.InteropServices.Marshal]::SizeOf($memory)
    if (-not [Wardrowbe.MemoryNative]::GlobalMemoryStatusEx([ref]$memory)) {
        throw [ComponentModel.Win32Exception]::new([Runtime.InteropServices.Marshal]::GetLastWin32Error())
    }
    if ($memory.TotalPhysical -eq 0 -or $memory.AvailablePhysical -gt $memory.TotalPhysical -or $memory.MemoryLoad -gt 100) {
        throw [IO.InvalidDataException]::new('Invalid native memory counters.')
    }
    return $memory
}

try {
    Write-MonitorState
    Initialize-NativeMemory
    while (-not (Test-Path -LiteralPath $StopPath)) {
        $memory = Read-NativeMemory
        $sample = @{ unix_seconds = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds(); available_bytes = [int64]$memory.AvailablePhysical }
        $temporary = $OutputPath + '.next'
        [IO.File]::WriteAllText($temporary, ($sample | ConvertTo-Json -Compress), [Text.UTF8Encoding]::new($false))
        if ([IO.File]::Exists($OutputPath)) { [IO.File]::Replace($temporary, $OutputPath, [System.Management.Automation.Language.NullString]::Value) }
        else { [IO.File]::Move($temporary, $OutputPath) }
        $monitorState.state = 'running'
        $monitorState.sample_count++
        $monitorState.last_success_unix_seconds = $sample.unix_seconds
        $monitorState.total_physical_bytes = [int64]$memory.TotalPhysical
        $monitorState.available_physical_bytes = [int64]$memory.AvailablePhysical
        $monitorState.memory_load_percent = $memory.MemoryLoad
        Write-MonitorState
        if ($Once) { break }
        Start-Sleep -Seconds 2
    }
    $monitorState.state = 'stopped'
    $monitorState.exit_reason = $(if ($Once) { 'single_sample_complete' } else { 'stop_marker' })
    Write-MonitorState
} catch {
    $monitorState.state = 'failed'
    $monitorState.exit_reason = 'sampling_error'
    $monitorState.error_type = $_.Exception.GetType().FullName
    $monitorState.error_hresult = $_.Exception.HResult
    $monitorState.error_message = $_.Exception.Message
    $monitorState.error_id = $_.FullyQualifiedErrorId
    $monitorState.error_stack = $_.ScriptStackTrace
    $monitorState.error_line = $_.InvocationInfo.ScriptLineNumber
    try { Write-MonitorState } catch {}
    [Console]::Error.WriteLine('wardrowbe_memory_monitor_failed: ' + $monitorState.error_type + '; hresult=' + $monitorState.error_hresult)
    [Console]::Error.WriteLine(($monitorState | ConvertTo-Json -Compress))
    exit 1
}
