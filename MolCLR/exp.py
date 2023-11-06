import os
import torch
from models.ginet_finetune import GINet
import pandas as pd
import torch.nn.functional as F
smiles_data = pd.read_csv("/home/luli/MolCLR/output_smiles.csv")
# smiles_list = smiles_data['Isomeric_SMILES'].tolist()
from dataset.dataset import MoleculeDatasetWrapper
torch.backends.cudnn.deterministic = True
dataset = MoleculeDatasetWrapper(6,4,0.1, "/home/luli/MolCLR/output_smiles.csv")
model = GINet( "classification" ).to("cuda:0")
checkpoints_folder = os.path.join('MolCLR/ckpt', "pretrained_gin", 'checkpoints')
state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location= "cuda:0")
model.load_my_state_dict(state_dict)
print("Loaded pre-trained model with success.")

representations = []
train_loader, valid_loader = dataset.get_data_loaders()
for bn, xis in enumerate(train_loader):
        print("i")
        xis = xis.to("cuda:0")
        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]
        # # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        # zjs = F.normalize(zjs, dim=1)
        print(xis)

 