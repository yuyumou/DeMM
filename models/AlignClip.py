import torch
import torch.nn as nn
import torch.nn.functional as F

class AlignCLIPSemanticLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, temperature=0.07):
        super().__init__()
        # self.alpha = alpha
        # self.beta = beta
        self.temperature = temperature
        init_logit = torch.logit(
            torch.tensor(0.5, dtype=torch.float32)
        )

        self.logit_alpha = nn.Parameter(init_logit)

    def forward(self, img_embeds, mol_embeds):
        # normalize
        img_embeds = F.normalize(img_embeds, dim=-1)
        mol_embeds = F.normalize(mol_embeds, dim=-1)

        # similarity matrices
        sim_i_t = torch.matmul(img_embeds, mol_embeds.t())   # cross-modal
        sim_i_i = torch.matmul(img_embeds, img_embeds.t())   # image intra-modal
        sim_t_t = torch.matmul(mol_embeds, mol_embeds.t())   # molecule intra-modal

        # semantic distance (D_y = 1 - S*S^T)
        D_y = 1 - sim_t_t

        # apply semantic-guided separation (Eq. 11)
        logits_vsep = sim_i_i * D_y + torch.eye(sim_i_i.shape[0], device=img_embeds.device)
        logits_vsep = logits_vsep / self.temperature

        # cross-modality logits (Eq. 12)
        logits_img_text = sim_i_t / self.temperature
        logits_text_img = sim_i_t.t() / self.temperature

        labels = torch.arange(img_embeds.size(0), device=img_embeds.device)

        # IMSep loss (semantic-guided intra-modal)
        loss_imsep = F.cross_entropy(logits_vsep, labels)

        # CRSep loss (cross-modal alignment)
        loss_crsep = 0.5 * (
            F.cross_entropy(logits_img_text, labels) +
            F.cross_entropy(logits_text_img, labels)
        )

        # alpha = torch.sigmoid(self.logit_alpha)
        beta  = 1.0 - alpha

        total_loss = alpha * loss_crsep + beta * loss_imsep

        # total loss (Eq. 13)
        # total_loss = self.alpha * loss_crsep + self.beta * loss_imsep

        # ablation study
        # total_loss =  loss_imsep 

        return total_loss, {
            "loss_crsep": loss_crsep.item(),
            "loss_imsep": loss_imsep.item()
        }
