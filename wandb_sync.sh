. venv_setup/activate.sh
echo "Syncing:"
while true
do 
    wandb sync logs/wandb/latest-run
    echo "..."
    sleep 60
done