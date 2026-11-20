<#
.SYNOPSIS
  Opens the report built by build_bmu_report.py in Word, updates every field (the table of contents, the
  page references in the lists of figures and tables) and saves it. Optionally writes each page as an EMF
  file for a visual check.

.EXAMPLE
  powershell -File scripts/finish_bmu_report.ps1 -Docx docs\report\Nightingale-Project-Report-BMU.docx
#>
param(
    [Parameter(Mandatory = $true)][string]$Docx,
    [string]$RenderDir = "",
    [string]$Pdf = "",
    [int]$MaxPages = 150
)
$ErrorActionPreference = "Stop"
$Docx = (Resolve-Path $Docx).Path
$w = New-Object -ComObject Word.Application
$w.Visible = $false
$w.DisplayAlerts = 0
try {
    $doc = $w.Documents.Open($Docx)
    $doc.Repaginate()
    for ($pass = 1; $pass -le 2; $pass++) {
        [void]$doc.Fields.Update()
        foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
        $doc.Repaginate()
    }
    $pages = $doc.ComputeStatistics(2)
    $words = $doc.ComputeStatistics(0)
    "pages: $pages  words: $words"
    $doc.Save()
    if ($Pdf -ne "") {
        $doc.ExportAsFixedFormat($Pdf, 17)   # wdExportFormatPDF
    }
    if ($RenderDir -ne "") {
        New-Item -ItemType Directory -Force -Path $RenderDir | Out-Null
        $pane = $w.ActiveWindow.ActivePane
        $count = [Math]::Min($pages, $MaxPages)
        for ($i = 1; $i -le $count; $i++) {
            $bits = $pane.Pages.Item($i).EnhMetaFileBits
            [System.IO.File]::WriteAllBytes((Join-Path $RenderDir ("page{0:D3}.emf" -f $i)), $bits)
        }
    }
    $doc.Close($false)
}
finally {
    $w.Quit()
}
