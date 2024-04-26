import math
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from einops.layers.torch import Rearrange

from src.datasets.catalog import DATASET_DICT
from src.systems.pytorch import base_system


class NMAESystem(base_system.BaseSystem):
    '''System for Masked Autoencoding.

    Masks a given fraction of input patches/tokens.
    Objective is to reconstruct masked items.
    '''

    def __init__(self, config):
        super().__init__(config)

        self.ce = torch.nn.CrossEntropyLoss(reduction='none')
        self.dataset = DATASET_DICT[config.dataset.name]
        if hasattr(self.dataset, 'MAE_OUTPUT_SIZE'):
            mae_output_size = self.dataset.MAE_OUTPUT_SIZE
            self.predictor = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(self.model.emb_dim, mae_output_size))

        self.accuracy = torchmetrics.Accuracy()

        self.k_num = 1
        self.k_size = 3
        self.loc224 = self.get_local_index(196, self.k_size)
        self.lamda = 5e-5
        self.off_diag = True
        self.token_reg = False

    @staticmethod
    def get_local_index(N_patches, k_size):
        """
        Get the local neighborhood of each patch 
        """
        loc_weight = []
        w = torch.LongTensor(list(range(int(math.sqrt(N_patches)))))
        for i in range(N_patches):
            ix, iy = i//len(w), i%len(w)
            wx = torch.zeros(int(math.sqrt(N_patches)))
            wy = torch.zeros(int(math.sqrt(N_patches)))
            wx[ix]=1
            wy[iy]=1
            for j in range(1,int(k_size//2)+1):
                wx[(ix+j)%len(wx)]=1
                wx[(ix-j)%len(wx)]=1
                wy[(iy+j)%len(wy)]=1
                wy[(iy-j)%len(wy)]=1
            weight = (wy.unsqueeze(0)*wx.unsqueeze(1)).view(-1)
            weight[i] = 0
            loc_weight.append(weight.nonzero().squeeze())
        return torch.stack(loc_weight)
    
    def forward(self, x, prehead=False):
        return self.model.forward(x, prehead=prehead)

    def ssl_forward(self, batch):
        batch = batch[1:-1]  # Remove label.

        # Embed first. [batch_size, seq_len, emb_dim]
        embs = self.model.embed(batch)
        masked_embs, masked_neighbors, is_masked, indices_to_mask = self.mask_embeddings(embs)

        if self.config.dataset.name in ['mscoco']:
            embed_img = nn.Sequential(
                Rearrange(
                    'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.dataset.PATCH_SIZE[0], p2=self.dataset.PATCH_SIZE[1]
                )
            )
            target_img = embed_img(batch[0])
            target_text = batch[1]
            target = [target_img, target_text]
        elif self.config.dataset.name in ['librispeech', 'chexpert', 'imagenet', 'eurosat', 'pamap2', 'cifar10_small',
                                          'wafer']:
            embed2 = nn.Sequential(
                Rearrange(
                    'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.dataset.PATCH_SIZE[0], p2=self.dataset.PATCH_SIZE[1]
                )
            )
            target = embed2(batch[0])
        elif self.config.dataset.name in ['wikitext103', 'mc4', 'pfam', 'genomics', 'higgs']:
            target = batch[0]
        else:
            raise ValueError(f'Unimplemented MAE dataset={self.config.dataset}.')

        # We pass prepool=True because we want the embeddings for each token.
        latent_s = self.model.encode(masked_embs, prepool=True)
        latent_n = self.model.encode(masked_neighbors, prepool=True)
        return latent_s, latent_n, is_masked, target, indices_to_mask

    def training_step(self, batch, batch_idx):
        embs, neighbor, is_masked, target, indices_to_mask = self.ssl_forward(batch)
        loss = self.objective(embs, neighbor, is_masked, target, indices_to_mask)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        embs, neighbor, is_masked, target, indices_to_mask = self.ssl_forward(batch)
        loss = self.objective(embs, neighbor, is_masked, target, indices_to_mask)
        self.log('val_loss', loss.item(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    def normalize_feature(self, pred):
        mean = pred.mean(dim=-1, keepdim=True)
        var = pred.var(dim=-1, keepdim=True)
        pred = (pred - mean) / (var + 1e-6)**0.5
        return pred
    
    def corr_loss(self, preds, predt):
        N, D, K = preds.shape

        zs = self.normalize_feature(preds)
        zt = self.normalize_feature(predt)

        # correlation
        if not self.token_reg:
            corr = torch.einsum("bki, bkj -> bij", zs, zt).div(D).mean(dim=0)
            diag = torch.eye(K, device=corr.device)
        else:
            corr = torch.einsum("bik, bjk -> bij", zs, zt).div(K).mean(dim=0)
            diag = torch.eye(D, device=corr.device)

        # diagonal
        if not self.off_diag:
            corr_loss = corr[diag.bool()].sum()
        else:
            corr_loss = corr[~diag.bool()].sum()
        return corr_loss.mul(self.lamda)

    def objective_continuous(self, embs, is_masked, target, embs_have_cls):
        preds = self.predictor(embs)
        if embs_have_cls:
            preds = preds[:, 1:]  # Remove [CLS] token.
        target = target.reshape(preds.shape[0], preds.shape[1], -1)
        diff = target[is_masked] - preds[is_masked]

        loss = torch.norm(diff, p=2, dim=-1).mean()
        return loss
    
    def objective_nmae(self, embs, neighbor, is_masked, target, embs_have_cls):
        preds = self.predictor(embs)

        with torch.no_grad():
            npreds = self.predictor(neighbor)

        if embs_have_cls:
            # Remove [CLS] token.
            preds = preds[:, 1:] 
            npreds = npreds[:, 1:]

        target = target.reshape(preds.shape[0], preds.shape[1], -1)
        diff = target[is_masked] - preds[is_masked]

        # correlation
        corr_loss = self.corr_loss(preds, npreds)
        rec_loss = torch.norm(diff, p=2, dim=-1).mean()

        loss = rec_loss + corr_loss
        return loss

    def objective_tokens(self, embs, is_masked, target, embs_have_cls, emb_module_idx=0):
        if embs_have_cls:
            embs = embs[:, 1:]
        embs = embs[is_masked]
        mask_preds = torch.einsum('ne,ve->nv', embs, self.model.embed_modules[emb_module_idx].embed.weight)
        mask_targets = target[is_masked]
        loss = self.ce(mask_preds, mask_targets).mean()
        return loss

    def objective(self, embs, neighbor, is_masked, target, indices_to_mask):
        # TODO: generalize to do this automatically based on specs.

        # Multimodal (tokenized and continuous)
        if self.config.dataset.name in ['mscoco']:
            image_seq_len = target[0].size(1)
            # Don't include CLS token
            img_loss = self.objective_continuous(
                embs[:, 1:image_seq_len + 1], is_masked[:, :image_seq_len], target[0], embs_have_cls=False
            )
            text_loss = self.objective_tokens(
                embs[:, 1 + image_seq_len:],
                is_masked[:, image_seq_len:],
                target[1],
                embs_have_cls=False,
                emb_module_idx=1  # In MSCOCO text is second.
            )
            loss = (img_loss + text_loss) / 2
        elif hasattr(self.dataset, 'MAE_OUTPUT_SIZE'):
            # Continuous data.
            # loss = self.objective_continuous(embs, is_masked, target, embs_have_cls=True)
            loss = self.objective_nmae(embs, neighbor, is_masked, target, embs_have_cls=True)
        else:
            # Tokenized data.
            loss = self.objective_tokens(embs, is_masked, target, embs_have_cls=True)
        return loss
    
    def sim_patches(self, x):
        N, L, D = x.shape
        
        x_norm = nn.functional.normalize(x, dim=-1)
        sim_matrix = x_norm[:,self.loc224] @ x_norm.unsqueeze(2).transpose(-2,-1)
        top_idx = sim_matrix.squeeze().topk(k=self.k_num,dim=-1)[1].view(N, L, self.k_num, 1)

        x_loc = x[:,self.loc224]
        x_loc = torch.gather(x_loc, 1, top_idx.expand(-1, -1, -1, D))
        return x_loc
    
    def get_neighbor(self, xs):
        N, L, D = xs.shape

        # get the neighbor patches (for teacher input)
        with torch.no_grad():
            x_loc = self.sim_patches(xs)
        
        return x_loc.view(N, int(x_loc.size(1)*x_loc.size(2)), D)

    def mask_embeddings(self, embs):
        '''Masks a fraction of embeddings within each example.

        Args:
            embs: [batch_size, seq_len, emb_size] embeddings to mask

        Returns:
            masked_embs: [batch_size, seq_len, emb_size] embeddings
                with specified fraction fraction masked
            is_masked: [batch_size, seq_len] of ints indicating whether each token was masked or not

        '''
        num_to_mask = int(np.ceil(embs.size(1) * self.config.corruption_rate))

        # neighborhood
        embs_loc = self.get_neighbor(embs)

        # [batch_size, num_to_mask]
        indices_to_mask = torch.rand(embs.size(0), embs.size(1), device=embs.device).topk(dim=-1, k=num_to_mask).indices
        
        # [batch_size, num_to_mask, emb_size] (repeat along last dimension for torch.scatter)
        indices_to_mask_for_scatter = indices_to_mask.unsqueeze(-1).repeat(1, 1, embs.size(-1))

        zeros = torch.zeros_like(indices_to_mask_for_scatter, device=embs.device, dtype=embs.dtype)
        masked_embs = torch.scatter(embs, -2, indices_to_mask_for_scatter, zeros)

        # masked neighbors
        masked_neighbors = torch.scatter(embs_loc, -2, indices_to_mask_for_scatter, zeros)

        is_masked = torch.zeros(embs.size(0), embs.size(1), dtype=int, device=embs.device)
        is_masked = torch.scatter(is_masked, 1, indices_to_mask, 1)

        return masked_embs, masked_neighbors, is_masked.bool(), indices_to_mask
