import torch
import torch.optim as optim
from MolCLR.dataset.dataset import MoleculeDatasetWrapper
from torch.utils.data import DataLoader
import os
from MolCLR.models.ginet_finetune import GINet
from model import ContrastiveLearningWithBioBERT
from config import config as args
from utils import load_checkpoint
from transformers import BertTokenizer, BertModel

torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

state_dict = torch.load(
    os.path.join(args.model.checkpoints_folder, args.model.molecule_model_checkpoint),
    map_location=device,
)


# Test the DrugDataset
# For demonstration purposes, I'm using placeholder paths. Replace with your actual paths.

dataset = MoleculeDatasetWrapper(args.data.batch_size, 1, 0.1, args.data.data_csv_path)
dataloader = dataset.get_data_loaders()

# Load the molecule model
molecule_model = GINet("classification", pred_act="relu").to(device)
molecule_model.load_my_state_dict(state_dict)
# Load the tokenizer and model
text_tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
text_model = BertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1").to(device)

# Assuming you have already defined the molecule model and loaded the pretrained weights
model = ContrastiveLearningWithBioBERT(
    text_tokenizer, text_model, molecule_model, device
)

optimizer = optim.Adam(model.text_model.parameters(), lr=args.train.learning_rate)
if args.resume.molecule and os.path.isfile(args.resume.molecule):
    print(f"Resuming training from {args.resume.molecule}")
    model, start_epoch = load_checkpoint(args.resume.molecule, model, optimizer)
    print(f"Resumed from epoch {start_epoch}")

if args.resume.text and os.path.isfile(args.resume.text):
    print(f"Resuming training from {args.resume.text}")
    start_epoch = load_checkpoint(args.resume.text, model, optimizer)
    print(f"Resumed from epoch {start_epoch}")

# Training loop
for epoch in range(args.train.num_epochs):
    print("here1")
    epoch_loss = 0.0
    for bn, batch in enumerate(dataloader):
        # Zero the gradients
        optimizer.zero_grad()
        batch_molecules, batch_texts = batch["graphs"], batch["texts"]
        # Compute the model's loss
        loss = model(batch_texts, batch_molecules)
        print(loss)
        # Backpropagate the loss
        loss.backward()

        # Update the model's weights
        optimizer.step()

        # Accumulate the loss for monitoring
        epoch_loss += loss.item()

    # Print the average loss for the epoch
    print(
        f"Epoch {epoch+1}/{args.train.num_epochs}, Loss: {epoch_loss / len(dataloader)}"
    )
