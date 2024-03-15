import random
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import copy

from vit import Transformer


class MAE(nn.Module):
    def __init__(
        self,
        encoder,
        decoder_dim,
        masking_strategy,
        masking_ratio,
        decoder_depth=1,
        decoder_heads=8,
        decoder_dim_head=64,
    ):
        super().__init__()
        assert 0 < masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        assert masking_strategy in ['random', 'random-hand']
        
        self.masking_ratio = masking_ratio
        self.masking_strategy = masking_strategy
        
        # extract some hyperparameters and functions from mae_encoder
        self.encoder = encoder        
        n_nodes, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.node_to_emb = encoder.to_node_embedding

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = (
            nn.Linear(encoder_dim, decoder_dim)
            if encoder_dim != decoder_dim
            else nn.Identity()
        )
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
            mlp_dim=decoder_dim * 4,
        )
                
        self.decoder_pos_emb = nn.Embedding(n_nodes, decoder_dim)
        self.to_pred = nn.Linear(decoder_dim, encoder.node_dim)
        
        self.unm_enc_to_dec = (
            nn.Linear(encoder_dim, decoder_dim)
            if encoder_dim != decoder_dim
            else nn.Identity()
        )
        
        self.n_nodes = n_nodes
        self.encoder_dim = encoder_dim
        
    def change_masking_strategy(self, strategy):
        assert strategy in ['random', 'one-hand'], "masking strategy must be in {'none', 'random', 'fingers', 'tips', 'connex', 'mixte'}"
        self.masking_strategy = strategy
        
    def change_masking_ratio(self, ratio):
        assert 0 < ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = ratio
        
    def _generate_mask(self, tokens, strategy, adj_matrix):
        
        device = tokens.device
        batch, n_tokens, token_dim = tokens.shape
        
        if strategy == 'random':
            num_masked = int(self.masking_ratio * n_tokens)
            rand_indices = torch.rand(batch, n_tokens, device=device).argsort(dim=-1)
            masked_indices, unmasked_indices = (
                rand_indices[:, :num_masked],
                rand_indices[:, num_masked:],
            )
            
        elif strategy == 'random-hand':
            n_tokens_per_hand = n_tokens // 2
            num_masked = int(self.masking_ratio * n_tokens_per_hand)
            rand_indices_left_hand = torch.rand(batch, n_tokens_per_hand, device=device).argsort(dim=-1)
            rand_indices_right_hand = 14 + torch.rand(batch, n_tokens_per_hand, device=device).argsort(dim=-1)
            
            masked_indices_left_hand, unmasked_indices_left_hand = (
                rand_indices_left_hand[:, :num_masked],
                rand_indices_left_hand[:, num_masked:],
            )
            
            masked_indices_right_hand, unmasked_indices_right_hand = (
                rand_indices_right_hand[:, :num_masked],
                rand_indices_right_hand[:, num_masked:],
            )
            
            masked_indices = torch.cat((masked_indices_left_hand, masked_indices_right_hand), dim=-1)
            unmasked_indices = torch.cat((unmasked_indices_left_hand, unmasked_indices_right_hand), dim=-1)
            num_masked *= 2
            
        else:
            raise NotImplementedError

        return num_masked, masked_indices, unmasked_indices
        
    def forward(self, x, a):
        device = x.device
        
        # get patches
        batch, n_nodes, node_dim = x.shape

        # patch to mae_encoder tokens and add positions
        tokens = self.node_to_emb(x)
        tokens = tokens.permute(0, 2, 1)
        tokens = tokens + self.encoder.pos_embedding

        #generate mask
        num_masked, masked_indices, unmasked_indices = self._generate_mask(tokens, self.masking_strategy, a)

        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device=device)[:, None]
        unmasked_tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss
        masked_nodes = x[batch_range, masked_indices]
        
        unmasked_a = torch.empty((*unmasked_indices.shape, unmasked_indices.shape[1]), device=device)
        for i, idx in enumerate(unmasked_indices):
            unmasked_a[i] = a[i, idx.view(-1, 1), idx]
              
        encoded_tokens = self.encoder.transformer(unmasked_tokens, unmasked_a)
        # project mae_encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens
        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_tokens = repeat(self.mask_token, "d -> b n d", b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim=1)
        decoded_tokens = self.decoder(decoder_tokens, a)
        
        # splice out the mask tokens and project to coordinates
        mask_tokens = decoded_tokens[batch_range, masked_indices]
        masked_pred = self.to_pred(mask_tokens)
        
        # calculate reconstruction loss
        recon_loss = F.mse_loss(masked_pred, masked_nodes)

        return decoder_tokens, masked_nodes, masked_indices, masked_pred, recon_loss
            
    def inference(self, x, a):
        device = x.device
        len_x = len(x.shape)
        if len_x == 4:
            b, t, n, d = x.shape
            x = rearrange(x, 'b t n d -> (b t) n d') 
            a = a.repeat(t, 1, 1)

        with torch.no_grad():
            # get patches
            batch, n_nodes, node_dim = x.shape

            # patch to mae_encoder tokens and add positions
            tokens = self.node_to_emb(x)
            tokens = tokens.permute(0, 2, 1)
            tokens = tokens + self.encoder.pos_embedding

            encoded_tokens = self.encoder.transformer(tokens, a)
            encoded_tokens = self.enc_to_dec(encoded_tokens)
        
        if len_x == 4:
            encoded_tokens = rearrange(encoded_tokens, '(b t) n d -> b t n d', b=b, t=t)
        
        return encoded_tokens
