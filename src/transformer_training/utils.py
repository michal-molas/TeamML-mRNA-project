import torch


def load_pretrained_weights(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    print(
        f"Loaded {checkpoint_path}  missing={len(missing)}  unexpected={len(unexpected)}"
    )
