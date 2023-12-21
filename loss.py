from torch import nn
import torch
import torch.nn.functional as F
import numpy as np


def gather_features(**kwargs):
    pass


class ClipLoss(nn.Module):
    def __init__(
        self,
        device="cuda",
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
        mlp_loss=False,
        weight_loss_kappa=0,
    ):
        super(ClipLoss, self).__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.mlp_loss = mlp_loss
        self.weighted_loss = bool(weight_loss_kappa != 0)
        self.weight_loss_kappa = weight_loss_kappa
        # cache state
        self.prev_num_logits = 0
        self.labels = {}
        self.logit_scale_d = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_t = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # nn.init.constant_(self.self.logit_scale_d, np.log(1 / 0.07))
        # nn.init.constant_(self.self.logit_scale_t, np.log(1 / 0.07))
        self.device = device
        self.to(device)

    def forward(
        self,
        drug_features,
        text_features,
        drug_features_mlp=None,
        text_features_mlp=None,
    ):
        # print(self.logit_scale_t.is_leaf)
        # self.logit_scale_d = self.self.logit_scale_d
        # self.logit_scale_t = self.self.logit_scale_t
        if self.mlp_loss:
            if self.world_size > 1:
                (
                    all_drug_features,
                    all_text_features,
                    all_drug_features_mlp,
                    all_text_features_mlp,
                ) = gather_features(
                    drug_features=drug_features,
                    text_features=text_features,
                    drug_features_mlp=drug_features_mlp,
                    text_features_mlp=text_features_mlp,
                    local_loss=self.local_loss,
                    gather_with_grad=self.gather_with_grad,
                    rank=self.rank,
                    world_size=self.world_size,
                    use_horovod=self.use_horovod,
                    mlp_loss=self.mlp_loss,
                )
                if self.local_loss:
                    a_logits_per_drug = (
                        self.logit_scale_d * drug_features @ all_text_features_mlp.T
                    )
                    a_logits_per_text = (
                        self.logit_scale_d * text_features_mlp @ all_drug_features.T
                    )
                    t_logits_per_drug = (
                        self.logit_scale_t * drug_features_mlp @ all_text_features.T
                    )
                    t_logits_per_text = (
                        self.logit_scale_t * text_features @ all_drug_features_mlp.T
                    )
                else:  # self.local_loss = False
                    a_logits_per_drug = (
                        self.logit_scale_d * all_drug_features @ all_text_features_mlp.T
                    )
                    a_logits_per_text = a_logits_per_drug.T
                    t_logits_per_drug = (
                        self.logit_scale_t * all_drug_features_mlp @ all_text_features.T
                    )
                    t_logits_per_text = t_logits_per_drug.T
            else:  # self.world_size = 1
                a_logits_per_drug = (
                    self.logit_scale_d * drug_features @ text_features_mlp.T
                )
                a_logits_per_text = (
                    self.logit_scale_d * text_features_mlp @ drug_features.T
                )
                t_logits_per_drug = (
                    self.logit_scale_t * drug_features_mlp @ text_features.T
                )
                t_logits_per_text = (
                    self.logit_scale_t * text_features @ drug_features_mlp.T
                )

            # calculated ground-truth and cache if enabled
            num_logits = a_logits_per_drug.shape[0]
            if self.prev_num_logits != num_logits or self.device not in self.labels:
                labels = torch.arange(num_logits, device=self.device, dtype=torch.long)
                if self.world_size > 1 and self.local_loss:
                    labels = labels + num_logits * self.rank
                if self.cache_labels:
                    self.labels[self.device] = labels
                    self.prev_num_logits = num_logits
            else:  # self.prev_num_logits == num_logits and device in self.labels
                labels = self.labels[self.device]

            if not self.weighted_loss:
                total_loss = (
                    F.cross_entropy(a_logits_per_drug, labels)
                    + F.cross_entropy(a_logits_per_text, labels)
                    + F.cross_entropy(t_logits_per_drug, labels)
                    + F.cross_entropy(t_logits_per_text, labels)
                ) / 4
            else:  # self.weighted_loss = True
                drug_weight = (drug_features @ drug_features.T).detach()
                drug_weight = (
                    torch.exp(
                        torch.sum(drug_weight, axis=1)
                        / (self.weight_loss_kappa * len(drug_weight))
                    )
                ).detach()
                text_weight = (text_features @ text_features.T).detach()
                text_weight = (
                    torch.exp(
                        torch.sum(text_weight, axis=1)
                        / (self.weight_loss_kappa * len(text_features))
                    )
                ).detach()
                total_loss = (
                    F.cross_entropy(a_logits_per_drug, labels, weight=drug_weight)
                    + F.cross_entropy(a_logits_per_text, labels, weight=drug_weight)
                    + F.cross_entropy(t_logits_per_drug, labels, weight=text_weight)
                    + F.cross_entropy(t_logits_per_text, labels, weight=text_weight)
                ) / 4
        else:  # self.mlp_loss = False
            if self.world_size > 1:
                all_drug_features, all_text_features = gather_features(
                    drug_features=drug_features,
                    text_features=text_features,
                    local_loss=self.local_loss,
                    gather_with_grad=self.gather_with_grad,
                    rank=self.rank,
                    world_size=self.world_size,
                    use_horovod=self.use_horovod,
                    mlp_loss=self.mlp_loss,
                )
                print("here")
                # print(self.logit_scale_t.is_leaf)
                if self.local_loss:
                    logits_per_drug = (
                        self.logit_scale_d * drug_features @ all_text_features.T
                    )
                    logits_per_text = (
                        self.logit_scale_d * text_features @ all_drug_features.T
                    )
                else:
                    logits_per_drug = (
                        self.logit_scale_d * all_drug_features @ all_text_features.T
                    )
                    logits_per_text = logits_per_drug.T
            else:  # self.world_size = 1
                logits_per_drug = self.logit_scale_d * drug_features @ text_features.T
                logits_per_text = self.logit_scale_d * text_features @ drug_features.T

            # calculated ground-truth and cache if enabled
            num_logits = logits_per_drug.shape[0]
            if self.prev_num_logits != num_logits or self.device not in self.labels:
                labels = torch.arange(num_logits, device=self.device, dtype=torch.long)
                if self.world_size > 1 and self.local_loss:
                    labels = labels + num_logits * self.rank
                if self.cache_labels:
                    self.labels[self.device] = labels
                    self.prev_num_logits = num_logits
            else:  # self.prev_num_logits == num_logits and device in self.labels
                labels = self.labels[self.device]
            if not self.weighted_loss:
                total_loss = (
                    F.cross_entropy(logits_per_drug, labels)
                    + F.cross_entropy(logits_per_text, labels)
                ) / 2
            else:  # self.weighted_loss = True
                drug_weight = (all_drug_features @ all_drug_features.T).detach()
                drug_weight = (
                    torch.exp(
                        torch.sum(drug_weight, axis=1)
                        / (self.weight_loss_kappa * len(all_drug_features))
                    )
                ).detach()
                text_weight = (all_text_features @ all_text_features.T).detach()
                text_weight = (
                    torch.exp(
                        torch.sum(text_weight, axis=1)
                        / (self.weight_loss_kappa * len(all_text_features))
                    )
                ).detach()
                total_loss = (
                    F.cross_entropy(logits_per_drug, labels, weight=text_weight)
                    + F.cross_entropy(logits_per_text, labels, weight=drug_weight)
                ) / 2
        return total_loss
