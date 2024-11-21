function startenv {
    ..\cse140_venv\Scripts\Activate
    Set-Alias python3 python
}

function exitenv {
    deactivate
}

function rungame {
    python3 -m pacai.bin.pacman
}