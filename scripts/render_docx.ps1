param([string]$Src, [string]$Pdf)
$w = New-Object -ComObject Word.Application
$w.Visible = $false
try {
    $d = $w.Documents.Open($Src, $false, $true)
    $d.ExportAsFixedFormat($Pdf, 17)
    "pages: " + $d.ComputeStatistics(2)
    $d.Close($false)
} finally { $w.Quit() }
