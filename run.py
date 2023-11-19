import torch
import torch.optim as optim
from MolCLR.dataset.dataset import MoleculeDatasetWrapper
from torch.utils.data import DataLoader
import os
from MolCLR.models.ginet_finetune import GINet
from model import ContrastiveLearningWithBioBERT
from config import config as args
from utils import load_checkpoint

from models import GINet, BertMLPModel
from loss import ClipLoss

torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

state_dict = torch.load(
    os.path.join(
        args.model.MolCLR.pretrained_folder, args.model.MolCLR.pretrained_model
    ),
    map_location=device,
)


# Test the DrugDataset
# For demonstration purposes, I'm using placeholder paths. Replace with your actual paths.

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
text_model = BertMLPModel(args, device)
if args.resume.molecule and os.path.isfile(args.resume.molecule):
    print(f"Resuming training from {args.resume.molecule}")
    molecule_state_dict = torch.load(args.resume.molecule)
    molecule_model.load_state_dict(molecule_state_dict, strict=False)

if args.resume.text and os.path.isfile(args.resume.text):
    print(f"Resuming training from {args.resume.text}")
    text_state_dict = torch.load(args.resume.text)
    text_model.load_state_dict(text_state_dict, strict=False)

molecule_model.freeze_GIN(
)
text_model.freeze_bert()

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

# Training loop
for epoch in range(args.train.num_epochs):
    print("here1")
    epoch_loss = 0.0
    for bn, batch in enumerate(dataloader):
        # Zero the gradients
        text_head_optimizer.zero_grad()
        molecule_head_optimizer.zero_grad()
        logit_scale_optimizer.zero_grad()
        batch_molecules, batch_texts = batch["graphs"], batch["texts"]
        batch_molecule_feat = molecule_model(batch_molecules)
        batch_molecule_feat = batch_molecule_feat / batch_molecule_feat.norm(
            dim=1, keepdim=True
        )
        batch_text_feat = text_model(batch_texts)
        batch_text_feat = batch_text_feat / batch_text_feat.norm(dim=1, keepdim=True)
        # Compute the model's loss
        l = loss(batch_molecule_feat, batch_text_feat)
        print(l)
        # Backpropagate the loss
        l.backward()

        # Update the model's weights
        text_head_optimizer.step()
        molecule_head_optimizer.step()
        logit_scale_optimizer.step()

        # Accumulate the loss for monitoring
        epoch_loss += loss.item()

    # Print the average loss for the epoch
    print(
        f"Epoch {epoch+1}/{args.train.num_epochs}, Loss: {epoch_loss / len(dataloader)}"
    )