from performer_pytorch import SelfAttention
from performer_pytorch import performer_pytorch
from performer_pytorch.performer_pytorch import default, exists, rearrange, empty, softmax_kernel, generalized_kernel
import torch
from functools import partial
import logging
log = logging.getLogger(__name__)


############################################################
# OVERWRITE THE PERFORMER IMPLEMENTATION (PERFORMER_PYTORCH)


def _orthogonal_matrix_chunk(cols, device = None):
    unstructured_block = torch.randn((cols, cols), device = device)
    q, r = torch.linalg.qr(unstructured_block.cpu(), mode = 'complete') 
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()

#### OUR EDIT TO THE PACKAGE
# Overwrite the old Implementation of orthogonal matrix chunking (PyTorch issue)
performer_pytorch.orthogonal_matrix_chunk = _orthogonal_matrix_chunk

class CustomSelfAttention(SelfAttention):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            try:
                self.local_attn.rel_pos = None
            except:
                log.warning("No Local Attention")
        
        def _compute_approximate_attention(self, q, k, mask=None):
            """
            Compute approximate attention matrix from Q and K using Performer's kernel.
            
            Returns attention matrix of shape (batch, heads, seq_len, seq_len)
            """
            device = q.device
            projection_matrix = self.fast_attention.projection_matrix
            
            # Apply the same kernel transformation as FastAttention
            if self.fast_attention.no_projection:
                q_prime = q.softmax(dim=-1)
                k_prime = k.softmax(dim=-2)
            elif self.fast_attention.generalized_attention:
                create_kernel = partial(
                    generalized_kernel,
                    kernel_fn=self.fast_attention.kernel_fn,
                    projection_matrix=projection_matrix,
                    device=device
                )
                q_prime, k_prime = map(create_kernel, (q, k))
            else:
                # Default: softmax kernel
                create_kernel = partial(
                    softmax_kernel,
                    projection_matrix=projection_matrix,
                    device=device
                )
                q_prime = create_kernel(q, is_query=True)
                k_prime = create_kernel(k, is_query=False)
            
            # Compute approximate attention: A[i,j] = φ(q_i) · φ(k_j)
            # Shape: (batch, heads, seq_len, seq_len)
            attn_matrix = torch.einsum('bhnd,bhmd->bhnm', q_prime, k_prime)
            
            # Apply mask if provided
            if mask is not None:
                mask_expanded = mask[:, None, None, :]  # (batch, 1, 1, seq_len)
                attn_matrix = attn_matrix.masked_fill(~mask_expanded, 0.0)
            
            # Normalize rows to get proper probability distribution
            attn_matrix = attn_matrix / (attn_matrix.sum(dim=-1, keepdim=True) + 1e-8)
            
            return attn_matrix
        
        def _compute_importance_scores(self, attn_matrix, mask=None):
            """
            Compute token importance from attention matrix (column sum = attention received).
            
            Returns importance scores of shape (batch, seq_len), normalized to sum to 1.
            """
            # Column sum = how much attention each token RECEIVES from others
            importance = attn_matrix.sum(dim=-2)  # (batch, heads, seq_len)
            
            # Average across heads
            importance = importance.mean(dim=1)  # (batch, seq_len)
            
            # Apply mask and normalize
            if mask is not None:
                importance = importance.masked_fill(~mask, 0.0)
            
            importance = importance / (importance.sum(dim=-1, keepdim=True) + 1e-8)
            
            return importance
                
        def forward(self, x, pos_emb = None, context = None, mask = None, context_mask = None, 
                    pos = None, 
                    pos_projection: bool = False,
                    return_attention: bool = False, **kwargs):
            assert not exists(context), 'self attention should not receive context'
            b, n, _, h, gh = *x.shape, self.heads, self.global_heads

            cross_attend = False 

            context = default(context, x)
            context_mask = mask  # OUR EDIT: default(context_mask, mask) if not cross_attend else context_mask

            q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

            ##### OUR EDITS TO THE PACKAGE:
            #if exists(pos):
            #    if pos_projection:
            #           pos = 
            #        q = self.sum(q,self.to_pos(pos))
            #    else:
            #        q = self.sum(q,pos)


            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
            (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))
            attn_outs = []
            
            attn_matrix = None
            importance_scores = None

            if not empty(q):
                if exists(context_mask):
                    global_mask = context_mask[:, None, :, None]
                    v.masked_fill_(~global_mask, 0.)

                ## OUR EDITS TO THE PACKAGE
                #if exists(pos_emb) and not cross_attend:
                #    q, k = apply_rotary_pos_emb(q, k, pos_emb)

                out = self.fast_attention(q, k, v)
                attn_outs.append(out)
                
                # Compute approximate attention if requested
                if return_attention:
                    attn_matrix = self._compute_approximate_attention(q, k, mask)
                    importance_scores = self._compute_importance_scores(attn_matrix, mask)

            if not empty(lq):
                assert not cross_attend, 'local attention is not compatible with cross attention'
                out = self.local_attn(lq, lk, lv, input_mask = mask)
                attn_outs.append(out)

            out = torch.cat(attn_outs, dim = 1)
            out = rearrange(out, 'b h n d -> b n (h d)')
            out =  self.to_out(out) # (batch_size, seq_len, hidden_size)
            out = self.dropout(out)
            
            if return_attention:
                return out, {
                    'attention_matrix': attn_matrix,
                    'importance_scores': importance_scores
                }
            return out


def get_attention_weighted_embedding(
    output_embeddings: torch.Tensor,
    importance_scores: torch.Tensor,
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute attention-weighted sequence embedding.
    
    Parameters
    ----------
    output_embeddings : torch.Tensor
        Token embeddings from model output, shape (batch, seq_len, hidden_dim)
    importance_scores : torch.Tensor
        Token importance scores, shape (batch, seq_len)
    mask : torch.Tensor, optional
        Attention mask, shape (batch, seq_len)
        
    Returns
    -------
    torch.Tensor
        Sequence-level embeddings, shape (batch, hidden_dim)
    """
    if mask is not None:
        importance_scores = importance_scores.masked_fill(~mask, 0.0)
        importance_scores = importance_scores / (importance_scores.sum(dim=-1, keepdim=True) + 1e-8)
    
    weights = importance_scores.unsqueeze(-1)  # (batch, seq_len, 1)
    sequence_embedding = (output_embeddings * weights).sum(dim=1)  # (batch, hidden_dim)
    
    return sequence_embedding