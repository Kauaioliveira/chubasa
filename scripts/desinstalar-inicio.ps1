<#
.SYNOPSIS
    Remove a tarefa que inicia o Chubasa ao entrar no Windows.

.EXAMPLE
    .\scripts\desinstalar-inicio.ps1 -WhatIf
    .\scripts\desinstalar-inicio.ps1
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$NomeDaTarefa = "Chubasa"
)

$ErrorActionPreference = "Stop"

$tarefa = Get-ScheduledTask -TaskName $NomeDaTarefa -ErrorAction SilentlyContinue
if ($null -eq $tarefa) {
    Write-Host "Não existe tarefa '$NomeDaTarefa': nada a remover."
    return
}

if ($PSCmdlet.ShouldProcess($NomeDaTarefa, "Parar e remover tarefa agendada")) {
    Stop-ScheduledTask -TaskName $NomeDaTarefa -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $NomeDaTarefa -Confirm:$false
    Write-Host "Tarefa '$NomeDaTarefa' removida. O Chubasa não inicia mais sozinho."
}
