param(
    [int]$Loops = 10,
    [int]$StartLoop = 1,
    [int]$EndLoop = 0,
    [int]$RunsPerGate = 5,
    [double]$MaxRegressionPct = 2.0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Ensure-Dir([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Write-JsonFile([object]$Data, [string]$Path) {
    $json = $Data | ConvertTo-Json -Depth 100
    Set-Content -LiteralPath $Path -Value $json -Encoding UTF8
}

function New-MetricsRecordFromObject($obj, [string]$tag) {
    return [ordered]@{
        tag = $tag
        bench_perf_avg_ms = [double]$obj.bench_perf_avg_ms
        bench_tilemap_500k_ns = [double]$obj.bench_tilemap_500k_ns
        bench_regression_xlarge_auto_avg_ms = [double]$obj.bench_regression_xlarge_auto_avg_ms
        bench_regression_xlarge_single_avg_ms = [double]$obj.bench_regression_xlarge_single_avg_ms
        bench_kernel_scalar_avg_ms = [double]$obj.bench_kernel_scalar_avg_ms
        bench_kernel_avx2_avg_ms = [double]$obj.bench_kernel_avx2_avg_ms
    }
}

function Median([double[]]$arr) {
    $vals = @($arr)
    if ($vals.Count -eq 0) { return 0.0 }
    $sorted = @($vals | Sort-Object)
    $mid = [int]($sorted.Count / 2)
    if (($sorted.Count % 2) -eq 1) {
        return [double]$sorted[$mid]
    }
    return ([double]$sorted[$mid - 1] + [double]$sorted[$mid]) / 2.0
}

function Parse-JsonFromMixedLines([string[]]$lines) {
    foreach ($line in $lines) {
        $trimmed = $line.Trim()
        if ($trimmed.StartsWith('{') -and $trimmed.EndsWith('}')) {
            try {
                return ($trimmed | ConvertFrom-Json)
            }
            catch {
            }
        }
    }

    $text = ($lines -join "`n")
    $start = $text.IndexOf('{')
    $end = $text.LastIndexOf('}')
    if ($start -ge 0 -and $end -gt $start) {
        $jsonText = $text.Substring($start, $end - $start + 1)
        return ($jsonText | ConvertFrom-Json)
    }

    throw "Could not parse JSON payload from command output"
}

function Parse-BenchPerfJson([string[]]$lines) {
    return Parse-JsonFromMixedLines $lines
}

function Parse-BenchKernelJson([string[]]$lines) {
    return Parse-JsonFromMixedLines $lines
}

function Parse-BenchTilemapNs([string[]]$lines) {
    $target = $null
    foreach ($line in $lines) {
        if ($line -match "500k cells, spread=5000") {
            $target = $line
            break
        }
    }
    if ($null -eq $target) {
        throw "Could not parse 500k benchmark line from bench_tilemap"
    }
    if ($target -match "\(([0-9\.]+) ns/cell") {
        return [double]$Matches[1]
    }
    throw "Could not parse ns/cell from line: $target"
}

function Parse-BenchRegressionXlargeAutoAvg([string[]]$lines) {
    foreach ($line in $lines) {
        if ($line -match "^xlarge-dense\s+auto\s+\d+\s+[0-9\.]+\s+([0-9\.]+)\s+") {
            return [double]$Matches[1]
        }
    }
    throw "Could not parse xlarge-dense auto avg from bench_regression"
}

function Parse-BenchRegressionXlargeSingleAvg([string[]]$lines) {
    foreach ($line in $lines) {
        if ($line -match "^xlarge-dense\s+1\s+\d+\s+[0-9\.]+\s+([0-9\.]+)\s+") {
            return [double]$Matches[1]
        }
    }
    throw "Could not parse xlarge-dense single-thread avg from bench_regression"
}

function Run-Cmd([string]$cmd) {
    Write-Host "`n>> $cmd"
    $tempOut = [System.IO.Path]::GetTempFileName()
    $tempErr = [System.IO.Path]::GetTempFileName()
    try {
        $proc = Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile', '-Command', $cmd) -RedirectStandardOutput $tempOut -RedirectStandardError $tempErr -NoNewWindow -PassThru -Wait
        $stdout = @()
        $stderr = @()
        if (Test-Path -LiteralPath $tempOut) {
            $stdout = Get-Content -LiteralPath $tempOut
        }
        if (Test-Path -LiteralPath $tempErr) {
            $stderr = Get-Content -LiteralPath $tempErr
        }
        $all = @($stdout + $stderr)
        if ($proc.ExitCode -ne 0) {
            throw "Command failed: $cmd`n$($all -join "`n")"
        }
        return $all
    }
    finally {
        Remove-Item -LiteralPath $tempOut -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $tempErr -ErrorAction SilentlyContinue
    }
}

function Cargo-RunLines([string]$cmd) {
    $lines = Run-Cmd $cmd
    return $lines
}

function Collect-Metrics([string]$tag, [int]$runs) {
    $benchPerfAvg = @()
    $benchTileNs = @()
    $benchRegAuto = @()
    $benchRegSingle = @()
    $benchKernelScalar = @()
    $benchKernelAvx2 = @()

    for ($i = 0; $i -lt $runs; $i++) {
        $perfCmd = 'cargo run --release --bin bench_perf -- --size 2048 --density 0.42 --warmup 3 --iters 30 --json'
        $perfLines = Cargo-RunLines $perfCmd
        $perfJson = Parse-BenchPerfJson $perfLines
        $benchPerfAvg += [double]$perfJson.avg_ms

        $tileLines = Cargo-RunLines 'cargo run --release --bin bench_tilemap'
        $benchTileNs += (Parse-BenchTilemapNs $tileLines)

        $regLines = Cargo-RunLines 'cargo run --release --bin bench_regression'
        $benchRegAuto += (Parse-BenchRegressionXlargeAutoAvg $regLines)
        $benchRegSingle += (Parse-BenchRegressionXlargeSingleAvg $regLines)

        $kernelLines = Cargo-RunLines 'cargo run --release --bin bench_kernel -- --size 2048 --density 0.42 --warmup 3 --iters 30 --json'
        $kernelJson = Parse-BenchKernelJson $kernelLines
        $benchKernelScalar += [double]$kernelJson.scalar.avg_ms
        if ($kernelJson.avx2.supported -eq $true) {
            $benchKernelAvx2 += [double]$kernelJson.avx2.avg_ms
        }
    }

    $result = [ordered]@{
        tag = $tag
        bench_perf_avg_ms = (Median $benchPerfAvg)
        bench_tilemap_500k_ns = (Median $benchTileNs)
        bench_regression_xlarge_auto_avg_ms = (Median $benchRegAuto)
        bench_regression_xlarge_single_avg_ms = (Median $benchRegSingle)
        bench_kernel_scalar_avg_ms = (Median $benchKernelScalar)
        bench_kernel_avx2_avg_ms = if ($benchKernelAvx2.Count -gt 0) { (Median $benchKernelAvx2) } else { 0.0 }
    }
    return $result
}

function Regression-Pct([double]$baseline, [double]$candidate, [bool]$lowerIsBetter = $true) {
    if ($baseline -eq 0.0) { return 0.0 }
    if ($lowerIsBetter) {
        return (($candidate - $baseline) / $baseline) * 100.0
    }
    return (($baseline - $candidate) / $baseline) * 100.0
}

function Is-NonRegressing([hashtable]$baseline, [hashtable]$candidate, [double]$maxRegressPct) {
    $checks = @(
        @{ name = 'bench_perf_avg_ms'; lower = $true },
        @{ name = 'bench_tilemap_500k_ns'; lower = $true },
        @{ name = 'bench_regression_xlarge_auto_avg_ms'; lower = $true },
        @{ name = 'bench_regression_xlarge_single_avg_ms'; lower = $true },
        @{ name = 'bench_kernel_scalar_avg_ms'; lower = $true }
    )

    if ($candidate.bench_kernel_avx2_avg_ms -gt 0.0 -and $baseline.bench_kernel_avx2_avg_ms -gt 0.0) {
        $checks += @{ name = 'bench_kernel_avx2_avg_ms'; lower = $true }
    }

    foreach ($c in $checks) {
        $k = $c.name
        $reg = Regression-Pct -baseline ([double]$baseline[$k]) -candidate ([double]$candidate[$k]) -lowerIsBetter $c.lower
        if ($reg -gt $maxRegressPct) {
            return @{ ok = $false; metric = $k; regress_pct = $reg }
        }
    }

    return @{ ok = $true; metric = ''; regress_pct = 0.0 }
}

function Restore-Files([string[]]$paths) {
    if ($paths.Count -eq 0) { return }
    $joined = ($paths | ForEach-Object { "`"$_`"" }) -join ' '
    Run-Cmd "git checkout -- $joined" | Out-Null
}

Ensure-Dir "target/perf"

if ($EndLoop -le 0) {
    $EndLoop = $Loops
}
if ($StartLoop -lt 1) {
    throw "StartLoop must be >= 1"
}
if ($EndLoop -lt $StartLoop) {
    throw "EndLoop must be >= StartLoop"
}

$env:RUSTFLAGS = "-C target-cpu=native -C target-feature=+avx2,+bmi2,+popcnt,+lzcnt"

Run-Cmd 'cargo --version' | Out-Null

$baseline = $null
if ($StartLoop -eq 1) {
    $baseline = Collect-Metrics -tag 'loop0-baseline' -runs $RunsPerGate
    Write-JsonFile -Data $baseline -Path 'target/perf/loop0-baseline.json'
}
else {
    $prevLoopPath = "target/perf/loop$($StartLoop - 1).json"
    if (Test-Path -LiteralPath $prevLoopPath) {
        $prev = Get-Content -LiteralPath $prevLoopPath | ConvertFrom-Json
        $baseline = New-MetricsRecordFromObject -obj $prev.full.metrics -tag "resume-from-loop$($StartLoop - 1)"
    }
    elseif (Test-Path -LiteralPath 'target/perf/loop0-baseline.json') {
        $baseObj = Get-Content -LiteralPath 'target/perf/loop0-baseline.json' | ConvertFrom-Json
        $baseline = New-MetricsRecordFromObject -obj $baseObj -tag 'resume-loop0'
    }
    else {
        throw "Cannot resume: no prior baseline found."
    }
}

$stepFiles = @{
    tilemap = @(
        'src/turbolife/tilemap.rs',
        'src/turbolife/arena.rs',
        'src/turbolife/engine.rs',
        'src/turbolife/activity.rs'
    )
    parallel = @('src/turbolife/engine.rs', 'src/turbolife/activity.rs', 'src/turbolife/sync.rs')
    algorithm = @('src/turbolife/engine.rs', 'src/turbolife/activity.rs', 'src/turbolife/arena.rs')
    simd = @('src/turbolife/kernel.rs')
}

$report = @()

for ($loop = $StartLoop; $loop -le $EndLoop; $loop++) {
    $entry = [ordered]@{
        loop = $loop
        accepted = $true
        steps = @()
        baseline_before = $baseline
    }

    foreach ($step in @('tilemap', 'parallel', 'algorithm', 'simd')) {
        $pre = Collect-Metrics -tag "loop${loop}-${step}-pre" -runs $RunsPerGate
        $post = Collect-Metrics -tag "loop${loop}-${step}-post" -runs $RunsPerGate
        $check = Is-NonRegressing -baseline $pre -candidate $post -maxRegressPct $MaxRegressionPct

        $stepResult = [ordered]@{
            step = $step
            accepted = [bool]$check.ok
            metric = [string]$check.metric
            regress_pct = [double]$check.regress_pct
            pre = $pre
            post = $post
        }

        if (-not $check.ok) {
            Restore-Files -paths $stepFiles[$step]
            $entry.accepted = $false
        }

        $entry.steps += $stepResult
        Write-JsonFile -Data $stepResult -Path "target/perf/loop${loop}-${step}.json"
    }

    $full = Collect-Metrics -tag "loop${loop}-full" -runs $RunsPerGate
    $loopCheck = Is-NonRegressing -baseline $baseline -candidate $full -maxRegressPct $MaxRegressionPct
    $entry.full = [ordered]@{
        accepted = [bool]$loopCheck.ok
        metric = [string]$loopCheck.metric
        regress_pct = [double]$loopCheck.regress_pct
        metrics = $full
    }

    if ($loopCheck.ok) {
        $baseline = $full
    } else {
        $entry.accepted = $false
    }

    Write-JsonFile -Data $entry -Path "target/perf/loop${loop}.json"
    $report += $entry
}

Write-JsonFile -Data @{
    generated_at = (Get-Date).ToString('o')
    start_loop = $StartLoop
    end_loop = $EndLoop
    loops = $report
} -Path "target/perf/summary-$StartLoop-$EndLoop.json"
Write-Host "Performance loop run complete. Summary: target/perf/summary-$StartLoop-$EndLoop.json"
