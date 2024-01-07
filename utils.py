import torch
import numpy as np
import random
import time
from datetime import datetime
import os
import glob
import subprocess
import re
import torch.nn as nn
from copy import deepcopy
import wandb
import fsspec

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


def get_hex_time(ms=False):
    """
    Description:
        get the current time in the format "DD/MM/YY HH:MM:SS" and convert it to a hexadecimal string
    """
    if ms:
        # Get current time with microseconds
        current_time = datetime.now().strftime("%d/%m/%y %H:%M:%S.%f")

        # Convert the time string to a datetime object
        dt_object = datetime.strptime(current_time, "%d/%m/%y %H:%M:%S.%f")

        # Convert the datetime object to a Unix timestamp with microseconds
        unix_time_with_microseconds = dt_object.timestamp()

        # Convert the Unix timestamp to a hexadecimal string, slicing off the '0x' and the 'L' at the end if it exists
        hex_time = hex(int(unix_time_with_microseconds * 1e6))[2:]

    else:
        current_time = time.strftime("%d/%m/%y %H:%M:%S", time.localtime())
        # convert the timestamp string to a Unix timestamp
        unix_time = int(time.mktime(time.strptime(current_time, "%d/%m/%y %H:%M:%S")))

        # convert the Unix timestamp to a hexadecimal string
        hex_time = hex(unix_time)[2:]

    return hex_time


def hex_to_time(hex_time, ms=False):
    """
    input:
        hex_time: str
    description:
        convert a hexadecimal string to a timestamp string in the format "DD/MM/YY HH:MM:SS"
    """
    # convert the hexadecimal string to a Unix timestamp
    if ms:
        # Convert the hexadecimal string to a Unix timestamp including microseconds
        unix_time_with_microseconds = (
            int(hex_time, 16) / 1e6
        )  # Divide by 1e6 to convert microseconds to seconds

        # Convert the Unix timestamp to a datetime object
        dt_object = datetime.fromtimestamp(unix_time_with_microseconds)

        # Format the datetime object to a string including microseconds
        time_str = dt_object.strftime("%d/%m/%y %H:%M:%S.%f")

    else:
        unix_time = int(hex_time, 16)

        # convert the Unix timestamp to a timestamp string in the format "DD/MM/YY HH:MM:SS"
        time_str = time.strftime("%d/%m/%y %H:%M:%S", time.localtime(unix_time))

    return time_str


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]

def get_latest_checkpoint(path: str, remote: bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(
            ["aws", "s3", "ls", path + "/"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [
            os.path.join(path, x.split(" ")[-1])
            for x in result.stdout.decode().split("\n")[:-1]
        ]
    else:
        checkpoints = glob.glob(path + "**/*.pt", recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None

def eval_(checkpoint, dataloader, loss, device, molecule_model, text_model):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # molecule_model = checkpoint["molecule_model"].to(device)
    # text_model = checkpoint["text_model"].to(device)
    molecule_model_sd = checkpoint["molecule_model_state_dict"]
    text_model_sd = checkpoint["text_model_state_dict"]
    # if not args.distributed and next(iter(sd.items()))[0].startswith("module"):
    #     sd = {k[len("module.") :]: v for k, v in sd.items()}
    molecule_model.load_state_dict(molecule_model_sd)
    text_model.load_state_dict(text_model_sd)
    text_model.device = device
    orig_loss = deepcopy(loss)
    loss.load_state_dict(checkpoint["loss_sd"])
    # loss = loss.to(device)
    # loss.device = device
    epoch_loss = 0
    epoch_loss_ori = 0
    with torch.no_grad():
        for bn, batch in enumerate(dataloader):
            batch_molecules, batch_texts = batch["graph"], batch["text"]
            batch_molecules = batch_molecules.to(device)
            batch_molecule_feat = molecule_model(batch_molecules)
            batch_molecule_feat = batch_molecule_feat[1] / batch_molecule_feat[1].norm(
                dim=1, keepdim=True
            )
            batch_text_feat = text_model(batch_texts)
            batch_text_feat = batch_text_feat / batch_text_feat.norm(dim=1, keepdim=True)
            # Compute the model's loss
            l = loss(batch_molecule_feat, batch_text_feat)
            l_ori = orig_loss(batch_molecule_feat, batch_text_feat)
            # Accumulate the loss for monitoring
            epoch_loss += l.item()
            epoch_loss_ori += l_ori.item()
    return epoch_loss / len(dataloader), epoch_loss_ori / len(dataloader)

def print_weights_and_gradients(model: nn.Module, print_grad=False, print_value=False):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if print_value:
                print(f"Weights of {name}: \n {param.data}")
            else:
                print(f"Weights of {name}")
            if print_grad:
                if param.grad is not None:
                    print(f"Gradient of {name}: \n {param.grad}")
                else:
                    print(f"Gradient of {name}: Not computed yet or no gradient")

 
def pt_load(file_path, map_location=None):
    # if file_path.startswith("s3"):
    #     logging.info("Loading remote checkpoint, which may take a bit.")
    of = fsspec.open(file_path, "rb")
    with of as f:
        out = torch.load(f, map_location=map_location)
    return out



def save_tensor_heatmap(tensor, draw_label=True, file_path="heat_map.png"):
    """
    Saves a heatmap of the given tensor to a PNG file and draws a blue line on the diagonal for comparison.

    Parameters:
    tensor (torch.Tensor): The tensor to create a heatmap from.
    file_path (str): The file path where the heatmap PNG will be saved.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 20))
    plt.imshow(tensor, cmap='hot', interpolation='nearest')
    plt.colorbar()
    if draw_label:
        # Drawing a blue line on the diagonal
        plt.plot(np.arange(tensor.shape[0]), np.arange(tensor.shape[1]), color='blue', linewidth=1)

    plt.title('Heatmap of Tensor with Diagonal Line')
    plt.savefig(file_path)
    plt.close()

def count_rows_with_max_on_diagonal(tensor):
    """
    Counts the number of rows in the tensor where the diagonal element is the maximum in that row.

    Parameters:
    tensor (torch.Tensor): The tensor to analyze.

    Returns:
    int: The number of rows with the diagonal element being the maximum of that row.
    """
    # Extracting the diagonal elements
    diagonal_elements = tensor.diag()

    # Counting rows where diagonal element is the maximum in that row
    count = torch.sum(torch.eq(tensor.max(dim=1).values, diagonal_elements)).item()

    return count