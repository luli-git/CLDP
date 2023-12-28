### import statements
import pickle
from config import config as args
import logging

if args.wandb:
    import wandb
import torch
import torch.optim as optim
from MolCLR.dataset.dataset import MoleculeDatasetWrapper
import os
from copy import deepcopy
from utils import load_checkpoint, print_model_differences, get_hex_time, setup_seed, natural_key, get_latest_checkpoint
from models import GINet, BertMLPModel
from loss import ClipLoss
from training.scheduler import cosine_lr, const_lr, const_lr_cooldown
from tqdm import tqdm
from utils import eval_
resume_latest = args.resume == "latest"
log_base_path = os.path.join(args.logs, args.name)
args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
LATEST_CHECKPOINT_NAME = "epoch_latest.pt"
args.time_stamp = get_hex_time(ms=True)
if not os.path.exists(args.checkpoint_path):
    os.makedirs(args.checkpoint_path)
if resume_latest:
    resume_from = None
    checkpoint_path = args.checkpoint_path

    # Checking for existing checkpoint via master rank only. It is possible for
    # different rank processes to see different files if a shared file-system is under
    # stress, however it's very difficult to fully work around such situations.
    if args.train.save_most_recent:
        # if --save-most-recent flag is set, look for latest at a fixed filename
        resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
        if not os.path.exists(resume_from):
            # If no latest checkpoint has been saved yet, don't try to resume
            resume_from = None
    else:
        # otherwise, list checkpoint dir contents and pick the newest checkpoint
        resume_from = get_latest_checkpoint(
            checkpoint_path, remote=args.remote_sync is not None
        )
    if resume_from:
        logging.info(f"Found latest resume checkpoint at {resume_from}.")
    else:
        logging.info(f"No latest resume checkpoint found in {checkpoint_path}.")

    args.resume = resume_from


# Set the random seeds for reproducibility
torch.backends.cudnn.deterministic = True
# set random seed for numpy, torch, cuda, and cudnn, python random
setup_seed(args.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

state_dict = torch.load(
    os.path.join(
        args.model.MolCLR.pretrained_folder, args.model.MolCLR.pretrained_model
    ),
    map_location=device,
)

# Load the dataset
dataset = MoleculeDatasetWrapper(args.data.batch_size, 1, 0.1, args.data.data_csv_path)
dataloader = dataset.get_data_loaders()


# Load the molecule model
molecule_model = GINet(args).to(device)
not_loaded = molecule_model.load_my_state_dict(state_dict, freeze_loaded=True)
print(
    "The following parameters were not loaded from pretrained molecule model: ",
    not_loaded,
)
# Load the tokenizer and model
text_model = BertMLPModel(args, device).to(device)
# if args.resume.molecule and os.path.isfile(args.resume.molecule):
#     print(f"Resuming training from {args.resume.molecule}")
#     molecule_state_dict = torch.load(args.resume.molecule)
#     molecule_model.load_state_dict(molecule_state_dict, strict=False)

# if args.resume.text and os.path.isfile(args.resume.text):
#     print(f"Resuming training from {args.resume.text}")
#     text_state_dict = torch.load(args.resume.text)
#     text_model.load_state_dict(text_state_dict, strict=False)
# optionally resume from a checkpoint
start_epoch = 0

import fsspec


def pt_load(file_path, map_location=None):
    if file_path.startswith("s3"):
        logging.info("Loading remote checkpoint, which may take a bit.")
    of = fsspec.open(file_path, "rb")
    with of as f:
        out = torch.load(f, map_location=map_location)
    return out



loss = ClipLoss(device=device, mlp_loss=False)
text_head_optimizer = optim.Adam(
    text_model.parameters(), lr=args.train.text_learning_rate
)
molecule_head_optimizer = optim.Adam(
    molecule_model.parameters(), lr=args.train.molecule_learning_rate
)
logit_scale_optimizer = optim.Adam(
    [loss.logit_scale_d, loss.logit_scale_t], lr=args.train.logit_scale_learning_rate
)
if args.resume is not None:
    checkpoint = pt_load(args.resume, map_location="cpu")

    if "epoch" in checkpoint:
        # resuming a train checkpoint w/ epoch and optimizer state
        start_epoch = checkpoint["epoch"] 
        molecule_model_sd = checkpoint["molecule_model_state_dict"]
        text_model_sd = checkpoint["text_model_state_dict"]
        # if not args.distributed and next(iter(sd.items()))[0].startswith("module"):
        #     sd = {k[len("module.") :]: v for k, v in sd.items()}
        molecule_model.load_state_dict(molecule_model_sd)
        text_model.load_state_dict(text_model_sd)
        # # Load the entire model objects
        # molecule_model = checkpoint["molecule_model"].to(device)
        # text_model = checkpoint["text_model"].to(device)
        # text_head_optimizer = pickle.loads(checkpoint["text_head_optimizer"])
        # molecule_head_optimizer = pickle.loads(checkpoint["molecule_head_optimizer"])
        # loss = checkpoint["loss_module"].to(device)
        # logit_scale_optimizer = pickle.loads(checkpoint["logit_scale_optimizer"])
        # loss.logit_scale_d = checkpoint["loss_module"].logit_scale_d
        # loss.logit_scale_t = checkpoint["loss_module"].logit_scale_t

        # loss.load_state_dict(checkpoint["loss_sd"])
        # loss.logit_scale_t.load_state_dict(checkpoint["loss_logit_scale_t"])
        
        if molecule_head_optimizer is not None:
            molecule_head_optimizer.load_state_dict(
                checkpoint["molecule_head_optimizer_sd"]
            )
        if text_head_optimizer is not None:
            text_head_optimizer.load_state_dict(checkpoint["text_head_optimizer_sd"])
        if logit_scale_optimizer is not None:
            logit_scale_optimizer.load_state_dict(checkpoint["logit_scale_optimizer_sd"])
        logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
    # else:
    #     # loading a bare (model only) checkpoint for fine-tune or evaluation
    #     model.load_state_dict(checkpoint)
    #     logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

# Freeze the weights of the pretrained models
molecule_model.freeze_GIN()
text_model.freeze_bert()

if args.wandb:
    wandb.init(project=args.wandb_project_name)
    wandb.log(args)
    wandb.config.batch_size = args.data.batch_size
    wandb.config.text_learning_rate = args.train.text_learning_rate
    wandb.config.molecule_learning_rate = args.train.molecule_learning_rate
    wandb.config.logit_scale_learning_rate = args.train.logit_scale_learning_rate

# create scheduler if train
scheduler = None
if (
    text_head_optimizer is not None
    and molecule_head_optimizer is not None
    and logit_scale_optimizer is not None
):
    
    total_samples = len(dataset)
    batch_size = args.data.batch_size  # Replace with your actual batch size variable

    # Calculate the number of batches per epoch
    num_batches_per_epoch = total_samples // batch_size
    # if total_samples % batch_size != 0:
    #     num_batches_per_epoch += 1  # Account for the last smaller batch

    total_steps = (num_batches_per_epoch) * args.train.num_epochs  
    # total_steps = 640 * args.train.num_epochs 

    if args.lr_scheduler == "cosine":
        scheduler = cosine_lr(
            text_head_optimizer,
            args.train.text_learning_rate,
            args.warmup,
            total_steps,
            args.train.min_lr,
        )
        scheduler_m = cosine_lr(
            molecule_head_optimizer,
            args.train.molecule_learning_rate,
            args.warmup,
            total_steps,
            args.train.min_lr,
        )
        scheduler_l = cosine_lr(
            logit_scale_optimizer,
            args.train.logit_scale_learning_rate,
            args.warmup,
            total_steps,
            args.train.min_lr,
        )
    elif args.lr_scheduler == "const":
        scheduler = const_lr(
            text_head_optimizer, args.train.text_learning_rate, args.warmup, total_steps
        )
        scheduler_m = const_lr(
            molecule_head_optimizer,
            args.train.molecule_learning_rate,
            args.warmup,
            total_steps,
            args.train.min_lr,
        )
        scheduler_l = const_lr(
            logit_scale_optimizer,
            args.train.logit_scale_learning_rate,
            args.warmup,
            total_steps,
            args.train.min_lr,
        )

    elif args.lr_scheduler == "const-cooldown":
        assert (
            args.epochs_cooldown is not None
        ), "Please specify the number of cooldown epochs for this lr schedule."
        cooldown_steps = (
            args.data.batch_size // args.accum_freq
        ) * args.epochs_cooldown
        scheduler = const_lr_cooldown(
            text_head_optimizer,
            args.train.text_learning_rate,
            args.warmup,
            total_steps,
            cooldown_steps,
            args.lr_cooldown_power,
            args.lr_cooldown_end,
        )
        scheduler_m = const_lr_cooldown(
            molecule_head_optimizer,
            args.train.molecule_learning_rate,
            args.warmup,
            total_steps,
            cooldown_steps,
            args.lr_cooldown_power,
            args.lr_cooldown_end,
        )
        scheduler_l = const_lr_cooldown(
            logit_scale_optimizer,
            args.train.logit_scale_learning_rate,
            args.warmup,
            total_steps,
            cooldown_steps,
            args.lr_cooldown_power,
            args.lr_cooldown_end,
        )

original_model = molecule_model

# Training loop
for epoch in tqdm(range(start_epoch, args.train.num_epochs)):
    epoch_loss = 0.0
    # save model weights
    # if args.debug:
    #     if epoch == 0:
    #         molecule_model0 = deepcopy(molecule_model)
    #         text_model0 = deepcopy(text_model)
    #     if epoch == 1:
    #         print_model_differences(molecule_model0, molecule_model)
    #         # print_model_differences(text_model0, text_model)
    for bn, batch in enumerate(dataloader):
        i_accum = bn  # // args.accum_freq
        step = (num_batches_per_epoch) * epoch + i_accum

        # step = 640 * epoch + i_accum
        if not args.skip_scheduler:
            scheduler(step)
            scheduler_m(step)
            scheduler_l(step)


        # Zero the gradients
        text_head_optimizer.zero_grad()
        molecule_head_optimizer.zero_grad()
        logit_scale_optimizer.zero_grad()
        batch_molecules, batch_texts = batch["graph"], batch["text"]
        batch_molecules = batch_molecules.to(device)
        # batch_texts = batch_texts.to(device)

        batch_molecule_feat = molecule_model(batch_molecules)
        batch_molecule_feat = batch_molecule_feat[1] / batch_molecule_feat[1].norm(
            dim=1, keepdim=True
        )
        batch_text_feat = text_model(batch_texts)
        batch_text_feat = batch_text_feat / batch_text_feat.norm(dim=1, keepdim=True)
        # Compute the model's loss
        l = loss(batch_molecule_feat, batch_text_feat)



        # print(l)

        # Backpropagate the loss
        l.backward()

        # Update the model's weights
        text_head_optimizer.step()
        molecule_head_optimizer.step()
        logit_scale_optimizer.step()

        # Accumulate the loss for monitoring
        epoch_loss += l.item()

    if args.debug:
        model_save_path = "/u/tianyuzh/CLDP/logs/CLDP/molecule_model_params.pth"
        torch.save(molecule_model.state_dict(), model_save_path)
        print(f"Model parameters saved to {model_save_path}")

        # Save optimizer state
        optimizer_save_path = "/u/tianyuzh/CLDP/logs/CLDP/molecule_head_optimizer_state.pth"
        torch.save(molecule_head_optimizer.state_dict(), optimizer_save_path)
        print(f"Optimizer state saved to {optimizer_save_path}")
        for param_group in molecule_head_optimizer.param_groups:
            print(param_group["lr"])
    for (param_name, saved_tensor), (_, current_tensor) in zip(original_model.state_dict().items(), molecule_model.state_dict().items()):
        # print(saved_tensor.keys())
        # print(current_tensor.keys())
        if not torch.allclose(saved_tensor, current_tensor, atol=1e-6):
            print(f"Mismatch found in '{param_name}'")

    completed_epoch = epoch + 1
    # Saving checkpoints.
    if args.save_logs:
        checkpoint_dict = {
            "epoch": completed_epoch,
            "name": args.name,
            "molecule_model_state_dict": molecule_model.state_dict(),
            "text_model_state_dict": text_model.state_dict(),
            "molecule_model": molecule_model,
            "original_molecule_model": original_model,
            "loss_module": loss, 
            "loss_sd": loss.state_dict(),
            "text_model": text_model,
            "text_head_optimizer_sd": text_head_optimizer.state_dict(),
            "molecule_head_optimizer_sd": molecule_head_optimizer.state_dict(),
            "logit_scale_optimizer_sd": logit_scale_optimizer.state_dict(),
            "text_head_optimizer": pickle.dumps(text_head_optimizer),
            "molecule_head_optimizer": pickle.dumps(molecule_head_optimizer),
            "logit_scale_optimizer": pickle.dumps(logit_scale_optimizer),
            "step": step + 1
        }
        # if scaler is not None:
        #     checkpoint_dict["scaler"] = scaler.state_dict()

        if completed_epoch == args.train.num_epochs or (
            args.train.save_frequency > 0
            and (completed_epoch % args.train.save_frequency) == 0
        ):
            torch.save(
                checkpoint_dict,
                os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
            )
            checkpoint1000 = torch.load(os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"))
            eval_loss,  eval_loss_orig = eval_(checkpoint1000, dataloader, loss, "cuda", molecule_model, text_model )
            print(eval_loss, eval_loss_orig)

            # print(eval_loss)
            # checkpoint1 = torch.load("/u/tianyuzh/CLDP/saved_checkpoints/epoch_1000.pt")
            # eval_loss1, eval_loss_orig1 = eval_(checkpoint1 , dataloader, loss, "cuda", molecule_model, text_model )
            # print(eval_loss1, eval_loss_orig1) 
            # print(epoch_loss / len(dataloader))
            # assert abs(eval_loss - epoch_loss / len(dataloader)) < 1e-6
        if args.train.delete_previous_checkpoint:
            previous_checkpoint = os.path.join(
                args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt"
            )
            if os.path.exists(previous_checkpoint):
                os.remove(previous_checkpoint)

        if args.train.save_most_recent:
            # try not to corrupt the latest checkpoint if save fails
            tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
            latest_save_path = os.path.join(
                args.checkpoint_path, LATEST_CHECKPOINT_NAME
            )
            torch.save(checkpoint_dict, tmp_save_path)
            os.replace(tmp_save_path, latest_save_path)
    
    if args.wandb:
        wandb.log({"loss": epoch_loss / len(dataloader),
                  "learning_rate":  molecule_head_optimizer.param_groups[0]["lr"],
                  "text_learning_rate": text_head_optimizer.param_groups[0]["lr"],
                  "logit_scale_learning_rate": logit_scale_optimizer.param_groups[0]["lr"]})

    # Print the average loss for the epoch
    print(
        f"Epoch {epoch+1}/{args.train.num_epochs}, Loss: {epoch_loss / len(dataloader)}"
    )
