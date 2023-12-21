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


def print_model_differences(model1, model2, name_pass=None, mode="output"):
    """
    Print the differences in parameter values between two models.

    Args:
    model1 (torch.nn.Module): The first model.
    model2 (torch.nn.Module): The second model.

    Returns:
    None
    """
    if mode == "output":
        cases = [f"start_{model1.name}\n"]
    for (name1, param1), (name2, param2) in zip(
        model1.named_parameters(), model2.named_parameters()
    ):
        assert name1 == name2, "The two models have different parameter names."
        if name_pass is not None:
            if name_pass in name1:
                continue
        if mode == "stop":
            assert torch.equal(
                param1, param2
            ), f"Parameters in layer {name1} are different."
        elif mode == "print":
            if not torch.equal(param1, param2):
                print(f"Parameters in layer {name1} are different.")
        elif mode == "output":
            if not torch.equal(param1, param2):
                cases.append(name1)
        else:
            raise ValueError(f"Undefined mode {mode}.")
    if mode == "output":
        with open("model_differences.txt", "w") as f:
            f.write("\n".join(cases))
