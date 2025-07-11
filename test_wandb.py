import wandb
run = wandb.init(
    entity="umich_med",
    project="diffsbdd",
    name="smoke_test_run",
    reinit=True
)
print("Started", run.name)
wandb.finish()
