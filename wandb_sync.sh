. venv_setup/activate.sh
echo "Syncing:"
while true
do 
    wandb sync logs/wandb/offline-run-*
    echo "..."
    sleep 60
done