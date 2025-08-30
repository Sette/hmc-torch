#!/bin/bash

REMOTE_HOST="Kaggle"
REMOTE_PATH="/kaggle/working/results/"
LOCAL_PATH="results/results_kaggle"

# Rodar a cada 1 minuto (altere para o intervalo desejado)
INTERVAL=60

while true; do
    echo "Sincronizando arquivos em: $(date)"
    rsync -avz --ignore-existing "$REMOTE_HOST:$REMOTE_PATH" "$LOCAL_PATH"
    # Ctrl+C para interromper de forma limpa
    trap "echo; echo 'Encerrando...'; exit 0" SIGINT SIGTERM
    sleep $INTERVAL
done
