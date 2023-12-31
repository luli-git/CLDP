import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from transformers import BertTokenizer, BertModel


class GINEConv(MessagePassing):
    def __init__(self, emb_dim, num_bond_type=5, num_bond_direction=3):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim), nn.ReLU(), nn.Linear(2 * emb_dim, emb_dim)
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GINet(nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """

    def __init__(self, args):
        super(GINet, self).__init__()
        self.num_layer = args.model.MolCLR.num_layer
        self.emb_dim = args.model.MolCLR.emb_dim
        self.feat_dim = args.model.MolCLR.feat_dim
        self.drop_ratio = args.model.MolCLR.drop_ratio
        self.name = "GINet"
        self.x_embedding1 = nn.Embedding(
            args.model.MolCLR.num_atom_type, args.model.MolCLR.emb_dim
        )
        self.x_embedding2 = nn.Embedding(
            args.model.MolCLR.num_chirality_tag, args.model.MolCLR.emb_dim
        )
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(self.num_layer):
            self.gnns.append(GINEConv(self.emb_dim))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(self.num_layer):
            self.batch_norms.append(nn.BatchNorm1d(self.emb_dim))

        if args.model.MolCLR.pool == "mean":
            self.pool = global_mean_pool
        elif args.model.MolCLR.pool == "max":
            self.pool = global_max_pool
        elif args.model.MolCLR.pool == "add":
            self.pool = global_add_pool
        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        self.pred_n_layer = max(1, args.model.MolCLR.pred_n_layer)

        if args.model.MolCLR.pred_act == "relu":
            pred_head = [
                nn.Linear(self.feat_dim, self.feat_dim // 2),
                nn.ReLU(inplace=True),
            ]
            for _ in range(self.pred_n_layer - 1):
                pred_head.extend(
                    [
                        nn.Linear(self.feat_dim // 2, self.feat_dim // 2),
                        nn.ReLU(inplace=True),
                    ]
                )
        elif args.model.MolCLR.pred_act == "softplus":
            pred_head = [nn.Linear(self.feat_dim, self.feat_dim // 2), nn.Softplus()]
            for _ in range(self.pred_n_layer - 1):
                pred_head.extend(
                    [nn.Linear(self.feat_dim // 2, self.feat_dim // 2), nn.Softplus()]
                )
        else:
            raise ValueError("Undefined activation function")

        pred_head.append(nn.Linear(self.feat_dim // 2, args.model.MolCLR.out_dim))
        self.pred_head = nn.Sequential(*pred_head)

    def forward(self, data, **kwargs):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            # h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        h = self.pool(h, batch)
        h = self.feat_lin(h)
        proj = self.pred_head(h)
        return h, proj

    def freeze_GIN(self):
        for name, param in self.named_parameters():
            if "pred_head" not in name:
                param.requires_grad = False

    def load_my_state_dict(self, state_dict, freeze_loaded=True):
        not_loaded = []
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                not_loaded.append(name)
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
            if freeze_loaded:
                own_state[name].requires_grad = False
        return not_loaded


class BertMLPModel(nn.Module):
    def __init__(self, args, device):
        super(BertMLPModel, self).__init__()
        self.text_tokenizer = BertTokenizer.from_pretrained(args.model.text.tokenizer)
        self.text_model = BertModel.from_pretrained(args.model.text.pretrained_model)
        additional_hidden_size = args.model.text.additional_hidden_size
        out_features = args.model.text.out_features

        self.pred_head = nn.Sequential(
            nn.Linear(self.text_model.config.hidden_size, additional_hidden_size),
            nn.ReLU(),
            nn.Linear(additional_hidden_size, out_features),
        )
        self.device = device
        self.name = "BertMLPModel"

    def forward(self, text):
        # Tokenize the input text
        encoded_input = self.text_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)
        # Get the representation from the text model (bioBERT)
        text_features = self.text_model(**encoded_input).last_hidden_state.mean(dim=1)
        logits = self.pred_head(text_features)
        return logits

    def freeze_bert(self):
        for param in self.text_model.parameters():
            param.requires_grad = False
