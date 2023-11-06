import torch.nn as nn

import torch.nn.functional as F
import torch
import torch.optim as optim

# from dataset import DrugDataset
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from MolCLR.dataset.dataset import MoleculeDatasetWrapper
from MolCLR.models.ginet_finetune import GINet
import numpy as np


class ContrastiveLearningWithBioBERT(nn.Module):
    def __init__(self, text_tokenizer, text_model, molecule_model, device):
        super(ContrastiveLearningWithBioBERT, self).__init__()
        self.text_tokenizer = text_tokenizer
        self.text_model = text_model
        # Molecule model
        self.molecule_model = molecule_model
        self.device = device
        # Freeze the molecule model
        for param in self.molecule_model.parameters():
            param.requires_grad = False

    def encode_text(self, text):
        # Tokenize the input text
        text = text
        encoded_input = self.text_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)
        # Get the representation from the text model (bioBERT)
        text_features = self.text_model(**encoded_input).last_hidden_state.mean(dim=1)
        text_features = text_features
        return text_features

    def encode_molecule(self, molecule_graph):
        molecule_graph = molecule_graph.to(self.device)
        ris, zis = self.molecule_model(
            molecule_graph.x,
            molecule_graph.edge_index,
            molecule_graph.edge_attr,
            molecule_graph.batch,
        )
        return zis

    def forward(self, text, molecule_rep):
        text_features = self.encode_text(text)
        molecule_features = self.encode_molecule(molecule_rep).to(self.device)
        # normalized features
        molecule_features = molecule_features / molecule_features.norm(
            dim=1, keepdim=True
        )
        # normalized features
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # Assuming text_features and molecule_features are your two matrices of size [batch_size, feature_size]
        cosine_similarities = F.cosine_similarity(
            text_features.unsqueeze(1), molecule_features.unsqueeze(0), dim=2
        )

        # Since you want to minimize the distance between pairs and maximize the distance between mismatched pairs,
        # you can subtract the cosine similarities from 1 to get a distance metric
        distances = 1 - cosine_similarities

        # Example margin
        margin = 0.5

        # Diagonal entries are the distances between matching pairs (positive pairs)
        positive_pair_distances = torch.diag(distances)

        # We need to mask out the positive pairs and only keep the negative pairs
        eye_mask = torch.eye(distances.size(0)).bool().to(distances.device)
        negative_pair_distances = distances.masked_fill(eye_mask, float("inf"))

        # Get the closest negative distance for each positive pair
        closest_negative_distances, _ = torch.min(negative_pair_distances, dim=1)

        # Calculate the loss such that positive distances are smaller than the closest negative distance by at least the margin
        losses = F.relu(positive_pair_distances - closest_negative_distances + margin)
        loss = losses.mean()  # Taking the mean over the batch

        return loss
