<#
.SYNOPSIS
    Faz o Chubasa iniciar sozinho ao entrar no Windows (Agendador de Tarefas, "ao fazer logon").

.DESCRIPTION
    Cria a tarefa "Chubasa" para o usuário atual, rodando
    .venv\Scripts\pythonw.exe -m chubasa --bandeja (sem janela, com ícone na bandeja).
    Não precisa de administrador. Para ver o que seria feito sem mudar nada, use -WhatIf.
    Para remover: scripts\desinstalar-inicio.ps1

.EXAMPLE
    .\scripts\instalar-inicio.ps1 -WhatIf
    .\scripts\instalar-inicio.ps1
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$NomeDaTarefa = "Chubasa"
)

$ErrorActionPreference = "Stop"

$raiz = Split-Path -Parent $PSScriptRoot
$pythonw = Join-Path $raiz ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $pythonw)) {
    throw "Não achei $pythonw. Crie o ambiente primeiro: py -3.12 -m venv .venv; .venv\Scripts\python -m pip install -e ."
}
if (-not (Test-Path (Join-Path $raiz ".env"))) {
    throw "Não achei o .env em $raiz. Crie com: Copy-Item .env.example .env"
}

$usuario = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

$acao = New-ScheduledTaskAction -Execute $pythonw -Argument "-m chubasa --bandeja" -WorkingDirectory $raiz
$gatilho = New-ScheduledTaskTrigger -AtLogOn -User $usuario
$principal = New-ScheduledTaskPrincipal -UserId $usuario -LogonType Interactive -RunLevel Limited
$opcoes = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew

Write-Host "Tarefa:   $NomeDaTarefa (ao fazer logon de $usuario)"
Write-Host "Comando:  `"$pythonw`" -m chubasa --bandeja"
Write-Host "Pasta:    $raiz"

if ($PSCmdlet.ShouldProcess($NomeDaTarefa, "Registrar tarefa agendada")) {
    Register-ScheduledTask -TaskName $NomeDaTarefa -Action $acao -Trigger $gatilho -Principal $principal `
        -Settings $opcoes -Description "Assistente de voz Chubasa (bandeja)" -Force | Out-Null
    Write-Host "Pronto: o Chubasa vai iniciar no próximo logon. Para iniciar agora: Start-ScheduledTask -TaskName $NomeDaTarefa"
}
