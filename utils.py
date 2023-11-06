import torch


# Function to save checkpoint
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)


# Function to load checkpoint
def load_checkpoint(checkpoint_path, model, optimizer, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, checkpoint.get("epoch", 0)
