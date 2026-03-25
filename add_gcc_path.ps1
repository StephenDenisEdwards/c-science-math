$currentPath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
if ($currentPath -notlike '*msys64*') {
    [Environment]::SetEnvironmentVariable('Path', $currentPath + ';C:\msys64\mingw64\bin', 'Machine')
    Write-Host "Added C:\msys64\mingw64\bin to system PATH"
} else {
    Write-Host "Already in system PATH"
}
