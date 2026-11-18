$ErrorActionPreference = "Stop"
$docx = (Get-ChildItem -Path . -Recurse -Filter *.docx -File |
        Where-Object { $_.DirectoryName -like "*\report" -and $_.Name -notlike "~`$*" } |
        Select-Object -First 1).FullName
if (-not $docx) { throw "report docx not found" }
$root = Split-Path $docx
$pdf = [System.IO.Path]::ChangeExtension($docx, ".pdf")
Get-ChildItem -Path $root -Filter "~`$*" -Force | Remove-Item -Force -ErrorAction SilentlyContinue
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($docx, $false, $true)
    $doc.SaveAs([ref]$pdf, [ref]17)
    $pages = $doc.ComputeStatistics(2)
    $chars = $doc.ComputeStatistics(3)
    Write-Output "docx=$docx"
    Write-Output "pages=$pages chars=$chars"
    $doc.Close($false)
} finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
Get-Item $pdf | Select-Object Name, Length, LastWriteTime
