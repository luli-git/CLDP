import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# 1. Models for embeddings
class VisionModel(nn.Module):
    def __init__(self, embed_dim=512):
        super(VisionModel, self).__init__()
        # Using ResNet as a backbone
        self.resnet = resnet18 = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_dim)
    
    def forward(self, x):
        return F.normalize(self.resnet(x), dim=1)

class TextModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512):
        super(TextModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=6)
        self.transformer = nn.Transformer(embed_dim, nhead=8)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)

        return F.normalize(x.mean(dim=1), dim=1)  # Mean-pooling over tokens

# 2. Contrastive Loss
def contrastive_loss(image_embeddings, text_embeddings, temperature=0.07):
    logits = torch.mm(image_embeddings, text_embeddings.t()) / temperature
    labels = torch.arange(len(image_embeddings)).to(logits.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
    return loss

# Sample run
if __name__ == "__main__":
    # Instantiate models
    vision_model = VisionModel()
    text_model = TextModel(vocab_size=10000)  # Dummy vocab size
    
    # Example data
    dummy_images = torch.randn(32, 3, 224, 224)
    dummy_texts = torch.randint(0, 10000, (32, 20))  # 32 sequences of 20 tokens each
    
    # Get embeddings
    image_embeddings = vision_model(dummy_images)
    text_embeddings = text_model(dummy_texts)
    
    # Compute loss
    loss = contrastive_loss(image_embeddings, text_embeddings)
    print(loss)
