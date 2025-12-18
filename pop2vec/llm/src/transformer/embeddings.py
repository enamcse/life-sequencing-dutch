from turtle import forward
import torch.nn as nn
from torch.nn.utils import parametrize
import torch
from pop2vec.llm.src.transformer.transformer_utils import ScaleNorm, ReZero, Norm, Center
import logging

log = logging.getLogger(__name__)

class Embeddings(nn.Module):
    "Class for token, position, segment and backgound embedding."

    def __init__(self, hparams):
        super(Embeddings, self).__init__()
        d = 0.01
        embedding_size = hparams.hidden_size

        self.token = nn.Embedding(
            hparams.vocab_size,
            embedding_size,
            padding_idx=0
        )

        self.age = PositionalEmbedding(1, hparams.hidden_size, torch.cos)
        self.abspos = PositionalEmbedding(1, hparams.hidden_size, torch.sin)
        self.segment = nn.Embedding(4, hparams.hidden_size, padding_idx=0)

        nn.init.uniform_(self.token.weight, a=-d, b=d)
        nn.init.uniform_(self.segment.weight, a=-d, b=d)
        
        if hparams.parametrize_emb:
            try:
                self.parametrize(norm=hparams.norm_input_emb)
            except:
                log.info("(EMBEDDING) Normalisation hyperparameter is not found, set to FALSE")
                self.parametrize()

        self.res_age = ReZero(hparams.hidden_size, simple=True, fill=0)
        self.res_abs = ReZero(hparams.hidden_size, simple=True, fill=0)
        self.res_seg = ReZero(hparams.hidden_size, simple=True, fill=0)
        self.dropout = nn.Dropout(hparams.emb_dropout)

        # Add these lines for experiment flags (default: False)
        self.no_age_emb = getattr(hparams, "no_age_emb", False)
        self.no_date_emb = getattr(hparams, "no_date_emb", False)

    def parametrize(self, norm: bool = False):
        """Remove Mean from the Embedding Matrix (on each forward pass"""
        ignore_index = torch.LongTensor([0,4,5,6,7,8])
        parametrize.register_parametrization(self.token, "weight", Center(ignore_index = ignore_index, norm=norm))

    def reparametrization(self):
        parametrize.remove_parametrizations(self.token, "weight", leave_parametrized=False)


    def forward(self, tokens, position, age, segment):
        tokens = self.token(tokens)

        # Debug age tensor before processing
        try:
            if torch.isnan(age).any() or torch.isinf(age).any():
                print(f"WARNING: Invalid age values detected: nan={torch.isnan(age).sum()}, inf={torch.isinf(age).sum()}")
                age = torch.nan_to_num(age, nan=0.0, posinf=100.0, neginf=0.0)
            
            # Ensure age is in reasonable range
            age_clamped = torch.clamp(age, min=0, max=120)
            if not torch.equal(age, age_clamped):
                print(f"WARNING: Age values clamped from range [{age.min():.2f}, {age.max():.2f}] to [0, 120]")
                age = age_clamped
            
            pos = self.age(age.float().unsqueeze(-1))
            
        except Exception as e:
            print(f"ERROR in age embedding: {e}")
            print(f"age tensor: shape={age.shape}, device={age.device}, dtype={age.dtype}")
            print(f"age range: [{age.min():.2f}, {age.max():.2f}]")
            raise e
            
        if self.no_age_emb:
            pos.zero_()
        pos[:, :5] *= 0
        tokens = self.res_age(tokens, pos)

        pos = self.abspos(position.float().unsqueeze(-1))
        if self.no_date_emb:
            pos.zero_()
        pos[:, :5] *= 0
        tokens = self.res_abs(tokens, pos)

        if segment is not None:
            pos = self.segment(segment)
            tokens = self.res_seg(tokens, pos)
        
        return self.dropout(tokens), None

def t2v(tau, f, w, b, w0, b0, arg=None):
    """Time2Vec function"""
    try:
        # Device debugging and synchronization
        devices = {
            'tau': tau.device if hasattr(tau, 'device') else 'unknown',
            'w': w.device if hasattr(w, 'device') else 'unknown',
            'b': b.device if hasattr(b, 'device') else 'unknown',
            'w0': w0.device if hasattr(w0, 'device') else 'unknown', 
            'b0': b0.device if hasattr(b0, 'device') else 'unknown'
        }
        
        # Check for device mismatches
        unique_devices = set(devices.values())
        if len(unique_devices) > 1:
            print(f"WARNING: Device mismatch in t2v: {devices}")
            # Move all tensors to the same device as tau
            target_device = tau.device
            w = w.to(target_device)
            b = b.to(target_device) 
            w0 = w0.to(target_device)
            b0 = b0.to(target_device)
        
        # Check for NaN or inf values
        if torch.isnan(tau).any() or torch.isinf(tau).any():
            print(f"WARNING: Invalid values in tau: nan={torch.isnan(tau).sum()}, inf={torch.isinf(tau).sum()}")
            tau = torch.nan_to_num(tau, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Synchronize CUDA if using GPU
        if tau.is_cuda:
            torch.cuda.synchronize()
        
        if arg:
            v1 = f(torch.matmul(tau, w) + b, arg)
        else:
            v1 = f(torch.matmul(tau, w) + b)
        v2 = torch.matmul(tau, w0) + b0
        result = torch.cat([v1, v2], -1)
        
        # Final synchronization
        if result.is_cuda:
            torch.cuda.synchronize()
            
        return result
        
    except Exception as e:
        print(f"ERROR in t2v: {e}")
        print(f"tau shape: {tau.shape}, device: {tau.device}")
        print(f"w shape: {w.shape}, device: {w.device}")
        print(f"b shape: {b.shape}, device: {b.device}")
        raise e


class PositionalEmbedding(nn.Module):
    """Implementation of Time2Vec"""
    def __init__(self, in_features, out_features, f):
        super(PositionalEmbedding, self).__init__()

        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = f

        d = 0.01 
        nn.init.uniform_(self.w0, a=-d, b=d)
        nn.init.uniform_(self.b0, a=-d, b=d)
        nn.init.uniform_(self.w, a=-d, b=d)
        nn.init.uniform_(self.b, a=-d, b=d)

    def forward(self, tau):
        try:
            # Ensure tau is valid and on the right device
            if tau.numel() == 0:
                print("WARNING: Empty tau tensor in PositionalEmbedding")
                return torch.zeros(tau.shape[0], tau.shape[1], self.w.shape[1] + 1, 
                                 device=tau.device, dtype=tau.dtype)
            
            # Check for invalid values
            if torch.isnan(tau).any() or torch.isinf(tau).any():
                print(f"WARNING: Invalid values in tau input: nan={torch.isnan(tau).sum()}, inf={torch.isinf(tau).sum()}")
                tau = torch.nan_to_num(tau, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Ensure all parameters are on the same device as input
            if tau.device != self.w.device:
                print(f"WARNING: Device mismatch - tau: {tau.device}, parameters: {self.w.device}")
            
            return t2v(tau, self.f, self.w, self.b, self.w0, self.b0)
            
        except Exception as e:
            print(f"ERROR in PositionalEmbedding.forward: {e}")
            print(f"tau: shape={tau.shape}, device={tau.device}, dtype={tau.dtype}")
            print(f"parameters device: {self.w.device}")
            raise e


############################
#### TEST (DUMMY EMBEDDINGS)
############################

class Test_Embeddings(nn.Module):
    "Class for token, position, segment and backgound embedding."

    def __init__(self, hparams):
        super(Embeddings, self).__init__()
        d = 0.01
        embedding_size = hparams.hidden_size

        self.token = nn.Embedding(
            hparams.vocab_size,
            embedding_size,
            padding_idx=0,
            #scale_grad_by_freq=True,#False,
            max_norm=1,
            norm_type=2,
        )

        self.age = PositionalEmbedding(1, hparams.hidden_size, torch.cos)
        self.abspos = PositionalEmbedding(1, hparams.hidden_size, torch.sin)

        self.segment = nn.Embedding(4, hparams.hidden_size, padding_idx=0, max_norm=1, norm_type=2)

        nn.init.uniform_(self.token.weight, a=-d, b=d)
        nn.init.uniform_(self.segment.weight, a=-d, b=d)
        if hparams.center_emb:
            self.center_embeddings()

        self.res_age = ReZero(hparams.hidden_size, simple=True, fill=0)
        self.res_abs = ReZero(hparams.hidden_size, simple=True, fill=0)
        self.res_seg = ReZero(hparams.hidden_size, simple=True, fill=0)
        self.dropout = nn.Dropout(hparams.emb_dropout)

    def center_embeddings(self):
        """Remove Mean from the Embedding Matrix (on each forward pass"""
        parametrize.register_parametrization(self.token, "weight", Norm())

    def reparametrization(self):
        parametrize.remove_parametrizations(self.token, "weight", leave_parametrized=False)


    def forward(self, tokens, position, age, segment):
        """"""
        tokens = self.token(tokens)

        pos = self.age(age.float().unsqueeze(-1))
        pos[:, :5] *= 0
        tokens = self.res_age(tokens, pos)

        pos = self.abspos(position.float().unsqueeze(-1))
        pos[:, :5] *= 0
        tokens = self.res_abs(tokens, pos)

        if segment is not None:
            tokens = self.res_seg(tokens, self.segment(segment))
        
        return self.dropout(tokens), None