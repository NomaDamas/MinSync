$ErrorActionPreference = "Stop"
$version = if ($env:KIWI_VERSION) { $env:KIWI_VERSION } else { "0.23.2" }
$prefix = if ($env:KIWI_PREFIX) { $env:KIWI_PREFIX } else { "$env:LOCALAPPDATA\kiwi" }
$base = "https://github.com/bab2min/Kiwi/releases/download/v$version"
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("kiwi-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $tmp, $prefix -Force | Out-Null
try {
    $libraryArchive = Join-Path $tmp "kiwi.zip"
    $modelArchive = Join-Path $tmp "model.tgz"
    Invoke-WebRequest "$base/kiwi_win_x64_v$version.zip" -OutFile $libraryArchive
    Invoke-WebRequest "$base/kiwi_model_v${version}_base.tgz" -OutFile $modelArchive
    Expand-Archive $libraryArchive -DestinationPath $prefix -Force
    tar -xzf $modelArchive -C $prefix
} finally {
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
Write-Output "KIWI_LIBRARY_PATH=$prefix\lib\kiwi.dll"
Write-Output "KIWI_MODEL_PATH=$prefix\models\cong\base"
