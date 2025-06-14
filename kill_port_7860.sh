#!/bin/bash

PORT=7860

echo "Cerco processi sulla porta $PORT..."

# Trova il PID del processo che occupa la porta
PID=$(lsof -ti :$PORT)

if [ -z "$PID" ]; then
    echo "Nessun processo trovato sulla porta $PORT"
    exit 0
fi

echo "Processo trovato: PID $PID"
echo "Dettagli processo:"
ps -p $PID -o pid,ppid,cmd --no-headers

echo "Termino il processo..."
kill -TERM $PID

# Aspetta 2 secondi per dare tempo al processo di terminare
sleep 2

# Verifica se il processo è ancora attivo
if kill -0 $PID 2>/dev/null; then
    echo "Il processo non si è terminato, forzo la chiusura..."
    kill -KILL $PID
    sleep 1
fi

# Verifica finale
FINAL_PID=$(lsof -ti :$PORT)
if [ -z "$FINAL_PID" ]; then
    echo "✓ Porta $PORT liberata con successo"
else
    echo "✗ Errore: processo ancora attivo sulla porta $PORT"
    exit 1
fi