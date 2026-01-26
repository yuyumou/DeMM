import os
import torch
import torch.nn as nn

from peft import PeftModel, get_peft_model, LoraConfig
from models.FoundationModels import inf_encoder_factory
from models.genomic_snn import SNN


class CUCAMLP(nn.Module):
    def __init__(self, backbone, num_cls, hidden_dim, proj_dim, dropout=0.25, batch_norm=False, aux_output=250, embed_type=None):
        super(CUCAMLP, self).__init__()

        self.embed_type = embed_type

        backbone_embed_dim_dict = {"hoptimus0": 1536, "gigapath": 1536, 
                                   "virchow": 2560, "virchow2": 2560, 
                                   "uni_v1": 1024, "conch_v1": 512, "plip": 768,
                                   "phikon": 768, "ctranspath": 768,
                                   "resnet50": 512}
        backbone_out_embed_dim = backbone_embed_dim_dict[backbone]

        self.projector_head = nn.Sequential(
            nn.Linear(backbone_out_embed_dim, hidden_dim),  
            nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity(),
            nn.ReLU(),  # 激活函数
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity(),
            nn.ReLU(),  
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
            )
        
        self.regression_head = torch.nn.Linear(proj_dim, num_cls)

        if self.embed_type == "geneexp":
            self.snn_branch = SNN(input_dim=aux_output, model_size_omic='huge', n_classes=aux_output)
        elif self.embed_type == "genept":
            self.snn_branch = SNN(input_dim=1536, model_size_omic='huge', n_classes=aux_output) # genept embedding dims 1536
        else:
            pass

    def forward(self, x, **kwargs):
        proj_embed = self.projector_head(x)
        reg_pred = self.regression_head(proj_embed)
        
        if self.embed_type in ["geneexp", "genept"]: # gene expression embedding or genePT embedding modes
            if 'gene_exp' in kwargs and 'gene_embed' in kwargs:
                batch_embedding = torch.matmul(kwargs['gene_exp'], kwargs['gene_embed']) if kwargs['gene_embed'] is not None else kwargs['gene_exp']

                molecu_embed, reconstr_pred = self.snn_branch(batch_embedding)
                return proj_embed, reg_pred, molecu_embed, reconstr_pred
        
        if 'return_embed' in kwargs and kwargs['return_embed']:
            return proj_embed, reg_pred
        else:
            return reg_pred # no embedding or commonly when inference





class CUCA(nn.Module):
    r"""
    CUCA model
    Args:
        - backbone : str
            The name of the backbone model
        - num_cls : int
            The number of classes
        - hidden_dim : int
            The hidden dimension of the projector head
        - proj_dim : int
            The projection dimension
        - dropout : float
            The dropout rate
        - batch_norm : bool
            Whether to use batch normalization
        - embed_type : str
            The type of embedding, one of ["geneexp", "genept", "no"]
        - **LoraCfgParams : dict
            The parameters for PE
    """
    def __init__(self, backbone, num_cls, hidden_dim, proj_dim, dropout=0.25, batch_norm=False, aux_output=250, embed_type=None, **LoraCfgParams):
        super(CUCA, self).__init__()

        self.embed_type = embed_type

        weights_path = os.path.join("model_weights_pretrained", backbone)
        self.backbone_name = backbone
        self.backbone = inf_encoder_factory(backbone)(weights_path)

        backbone_out_embed_dim = self.backbone.out_embed_dim
        print(backbone_out_embed_dim)

        self.projector_head = nn.Sequential(
            nn.Linear(backbone_out_embed_dim, hidden_dim),  
            nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity(),
            nn.ReLU(),  # 激活函数
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity(),
            nn.ReLU(),  
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
            nn.BatchNorm1d(proj_dim) if batch_norm else nn.Identity(),
            nn.ReLU(),  
            )
        
        self.regression_head = torch.nn.Linear(proj_dim, num_cls)
        self.process_backbone(**LoraCfgParams)

        if proj_dim == 256:
            model_size_type = 'big'
        elif proj_dim == 512:
            model_size_type = 'huge'
        elif proj_dim == 1024:
            model_size_type = 'giant'
        else:
            raise NotImplementedError

        if self.embed_type == "geneexp":
            self.snn_branch = SNN(input_dim=aux_output, model_size_omic=model_size_type, n_classes=aux_output)
        elif self.embed_type == "genept":
            self.snn_branch = SNN(input_dim=1536, model_size_omic=model_size_type, n_classes=aux_output) # genept embedding dims 1536
        else:
            pass

    def process_backbone(self, **lora_cfg_kwargs):
        if lora_cfg_kwargs['ft_lora']:
            del lora_cfg_kwargs['ft_lora']
            if self.backbone_name == 'phikon' or self.backbone_name == 'phikon2':
                for name, module in self.backbone.encoder.named_modules(): # get the target modules in encoder
                    if name.split('.')[0] == 'encoder':
                        if isinstance(module, torch.nn.Linear) and name.split('.')[2] in lora_cfg_kwargs['only_spec_blocks']:
                            lora_cfg_kwargs['target_modules'].append(name)
            else:
                for name, module in self.backbone.encoder.named_modules(): # get the target modules in encoder
                    if isinstance(module, torch.nn.Linear) and name.split('.')[1] in lora_cfg_kwargs['only_spec_blocks']:
                        lora_cfg_kwargs['target_modules'].append(name)         
            del lora_cfg_kwargs['only_spec_blocks']

            lora_config = LoraConfig(**lora_cfg_kwargs)
            self.backbone.encoder = get_peft_model(self.backbone.encoder, lora_config) 
        else: # no lora, fixed
            for name, param in self.backbone.encoder.named_parameters():
                param.requires_grad = False

    def forward(self, x, **kwargs):
        embedding = self.backbone(x)
        proj_embed = self.projector_head(embedding)
        reg_pred = self.regression_head(proj_embed)
        
        if self.embed_type in ["geneexp", "genept"]: # gene expression embedding or genePT embedding modes
            if 'gene_exp' in kwargs and 'gene_embed' in kwargs:
                batch_embedding = torch.matmul(kwargs['gene_exp'], kwargs['gene_embed']) if kwargs['gene_embed'] is not None else kwargs['gene_exp']

                molecu_embed, reconstr_pred = self.snn_branch(batch_embedding)
                return proj_embed, reg_pred, molecu_embed, reconstr_pred
        
        if 'return_embed' in kwargs and kwargs['return_embed']:
            return proj_embed, reg_pred
        else:
            return reg_pred # no embedding or commonly when inference



from AlignClip import AlignCLIPSemanticLoss


class DeMM(nn.Module):
    r"""
    CUCA model
    Args:
        - backbone : str
            The name of the backbone model
        - num_cls : int
            The number of classes
        - hidden_dim : int
            The hidden dimension of the projector head
        - proj_dim : int
            The projection dimension
        - dropout : float
            The dropout rate
        - batch_norm : bool
            Whether to use batch normalization
        - embed_type : str
            The type of embedding, one of ["geneexp", "genept", "no"]
        - **LoraCfgParams : dict
            The parameters for PE
    """
    def __init__(self, backbone, num_cls, hidden_dim, proj_dim, dropout=0.25, batch_norm=False, aux_output=250, embed_type=None, **LoraCfgParams):
        super(DeMM, self).__init__()

        self.align_loss_fn = AlignCLIPSemanticLoss(
            temperature=0.07
        )


        self.embed_type = embed_type

        weights_path = os.path.join("model_weights_pretrained", backbone)
        self.backbone_name = backbone
        self.backbone = inf_encoder_factory(backbone)(weights_path)

        backbone_out_embed_dim = self.backbone.out_embed_dim
        print(backbone_out_embed_dim)

        self.projector_head = nn.Sequential(
            nn.Linear(backbone_out_embed_dim, hidden_dim),  
            nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity(),
            nn.ReLU(),  # 激活函数
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity(),
            nn.ReLU(),  
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
            nn.BatchNorm1d(proj_dim) if batch_norm else nn.Identity(),
            nn.ReLU(),  
            )
        
        self.regression_head = torch.nn.Linear(proj_dim, num_cls)
        self.process_backbone(**LoraCfgParams)

        if proj_dim == 256:
            model_size_type = 'big'
        elif proj_dim == 512:
            model_size_type = 'huge'
        elif proj_dim == 1024:
            model_size_type = 'giant'
        else:
            raise NotImplementedError

        if self.embed_type == "geneexp":
            self.snn_branch = SNN(input_dim=aux_output, model_size_omic=model_size_type, n_classes=aux_output)
        elif self.embed_type == "genept":
            self.snn_branch = SNN(input_dim=1536, model_size_omic=model_size_type, n_classes=aux_output) # genept embedding dims 1536
        else:
            pass

    def process_backbone(self, **lora_cfg_kwargs):
        if lora_cfg_kwargs['ft_lora']:
            del lora_cfg_kwargs['ft_lora']

            if self.backbone_name == "phikon" or self.backbone_name == "phikon2":
                
                for name, module in self.backbone.encoder.named_modules(): # get the target modules in encoder

                    if name.split('.')[0] == 'encoder':
                        if isinstance(module, torch.nn.Linear) and name.split('.')[2] in lora_cfg_kwargs['only_spec_blocks']:
                            lora_cfg_kwargs['target_modules'].append(name)
            else:
                for name, module in self.backbone.encoder.named_modules(): # get the target modules in encoder
                    if isinstance(module, torch.nn.Linear) and name.split('.')[1] in lora_cfg_kwargs['only_spec_blocks']:
                        lora_cfg_kwargs['target_modules'].append(name)         
            
            del lora_cfg_kwargs['only_spec_blocks']

            lora_config = LoraConfig(**lora_cfg_kwargs)
            self.backbone.encoder = get_peft_model(self.backbone.encoder, lora_config) 
        else: # no lora, fixed
            for name, param in self.backbone.encoder.named_parameters():
                param.requires_grad = False

    def forward(self, x, **kwargs):
        embedding = self.backbone(x)
        proj_embed = self.projector_head(embedding)
        reg_pred = self.regression_head(proj_embed)
        
        if self.embed_type in ["geneexp", "genept"]: # gene expression embedding or genePT embedding modes
            if 'gene_exp' in kwargs and 'gene_embed' in kwargs:
                batch_embedding = torch.matmul(kwargs['gene_exp'], kwargs['gene_embed']) if kwargs['gene_embed'] is not None else kwargs['gene_exp']

                molecu_embed, reconstr_pred = self.snn_branch(batch_embedding)
                loss_align, loss_info = self.align_loss_fn(
                    proj_embed, molecu_embed
                )

                return proj_embed, reg_pred, molecu_embed, reconstr_pred,loss_align
        
        if 'return_embed' in kwargs and kwargs['return_embed']:
            return proj_embed, reg_pred
        else:
            return reg_pred # no embedding or commonly when inference


class NoisyGating(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, dropout=0.1):
        super(NoisyGating, self).__init__()
        self.num_experts = num_experts
        # Feature extraction layer
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Gating layer
        self.w_gate = nn.Linear(hidden_dim, num_experts)
        # Noise layer
        self.w_noise = nn.Linear(hidden_dim, num_experts)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        features = self.layer1(x)
        clean_logits = self.w_gate(features)
        
        if self.training:
            raw_noise_stddev = self.w_noise(features)
            noise_stddev = torch.nn.functional.softplus(raw_noise_stddev) + 1e-2
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        
        soft_gates = self.softmax(logits)
        
        # Load Balancing Loss
        if self.training:
            # Encouraging uniform distribution of experts across the batch
            # P_i = mean probability of expert i in the batch
            mean_gate_weights = torch.mean(soft_gates, dim=0)
            # Loss = sum(P_i^2) * N. Minimal when P_i = 1/N
            balance_loss = torch.sum(mean_gate_weights ** 2) * self.num_experts
        else:
            balance_loss = torch.tensor(0.0, device=x.device)
            
        return soft_gates, balance_loss


class MoE_DeMM(DeMM):
    r"""
    MoE DeMM model with Mixture of Experts for robust prediction
    Args:
        - num_experts: int
            Number of experts to use
    """
    def __init__(self, backbone, num_cls, hidden_dim, proj_dim, dropout=0.25, batch_norm=False, aux_output=250, embed_type=None, num_experts=4, gate_loss_weight=0.1, **LoraCfgParams):
        super(MoE_DeMM, self).__init__(backbone, num_cls, hidden_dim, proj_dim, dropout, batch_norm, aux_output, embed_type, **LoraCfgParams)
        
        # Override single heads with MoE components
        self.num_experts = num_experts
        self.gate_loss_weight = gate_loss_weight
        
        # We need the input dim for the heads, which is backbone_out_embed_dim
        # Since it's not stored in DeMM explicitly as a property, we re-access it from backbone
        backbone_out_embed_dim = self.backbone.out_embed_dim
        
        # Define Experts
        # Each expert mimics the original projector + regression head structure
        # Re-defining experts to split projector and regressor to allow access to projected embeddings
        self.expert_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(backbone_out_embed_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, proj_dim),
                nn.BatchNorm1d(proj_dim) if batch_norm else nn.Identity(),
                nn.ReLU(),
            ) for _ in range(num_experts)
        ])
        
        self.expert_regressors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(proj_dim, proj_dim),
                nn.BatchNorm1d(proj_dim) if batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(proj_dim, num_cls)
            ) for _ in range(num_experts)
        ])
        
        # Shared Expert (Always Active)
        # This acts as a "baseline" or "anchor" to ensure stability and capture common features
        self.shared_projector = nn.Sequential(
            nn.Linear(backbone_out_embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
            nn.BatchNorm1d(proj_dim) if batch_norm else nn.Identity(),
            nn.ReLU(),
        )
        
        self.shared_regressor = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_cls)
        )

        # Gating Network with Noisy Gating
        # Input: backbone embedding, Output: weights for each expert
        self.gate = NoisyGating(input_dim=backbone_out_embed_dim, hidden_dim=256, num_experts=num_experts, dropout=0.1)
        
        # Remove old single heads to save memory
        del self.projector_head
        del self.regression_head

    def forward(self, x, **kwargs):
        embedding = self.backbone(x) # [B, D_backbone]
        
        # 1. Shared Expert Forward
        shared_proj = self.shared_projector(embedding)
        shared_pred = self.shared_regressor(shared_proj)

        # 2. Gating
        gate_weights, gate_loss = self.gate(embedding) # [B, num_experts], scalar
        
        # 3. Experts Forward (Routed Experts)
        # We need to stack outputs to perform weighted sum
        
        # [B, num_experts, proj_dim]
        expert_proj_embeds = torch.stack([proj(embedding) for proj in self.expert_projectors], dim=1)
        
        # [B, num_experts, num_cls]
        expert_preds = torch.stack([reg(proj_embed) for reg, proj_embed in zip(self.expert_regressors, torch.unbind(expert_proj_embeds, dim=1))], dim=1)
        
        # 4. Weighted Sum
        # routed_proj: [B, proj_dim] - Weighted average of projected embeddings
        routed_proj = torch.einsum('be,bed->bd', gate_weights, expert_proj_embeds)
        
        # routed_pred: [B, num_cls] - Weighted average of predictions
        routed_pred = torch.einsum('be,bec->bc', gate_weights, expert_preds)
        
        # 5. Final Combination (Shared + Routed)
        proj_embed = shared_proj + routed_proj
        reg_pred = shared_pred + routed_pred

        # 6. DeMM Logic (Alignment)
        if self.embed_type in ["geneexp", "genept"]:
            if 'gene_exp' in kwargs and 'gene_embed' in kwargs:
                batch_embedding = torch.matmul(kwargs['gene_exp'], kwargs['gene_embed']) if kwargs['gene_embed'] is not None else kwargs['gene_exp']

                molecu_embed, reconstr_pred = self.snn_branch(batch_embedding)
                
                # Use the weighted proj_embed for alignment
                loss_align, loss_info = self.align_loss_fn(
                    proj_embed, molecu_embed
                )
                
                # Add gate loss to alignment loss to avoid changing return signature
                total_aux_loss = loss_align + self.gate_loss_weight * gate_loss
                
                return proj_embed, reg_pred, molecu_embed, reconstr_pred, total_aux_loss
        
        if 'return_embed' in kwargs and kwargs['return_embed']:
            return proj_embed, reg_pred
        else:
            return reg_pred
