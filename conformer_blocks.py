"""
Conformer Blocks for BAT-Frame
Boundary-Biased Attention Mechanism for Partial Spoofing Detection

Key Innovation: Inject boundary prior as soft attention bias instead of hard gating.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class FeedForwardModule(nn.Module):
    """
    Macaron-style Feed Forward Module.
    expansion_factor=4 follows the original Conformer paper.
    """
    def __init__(self, d_model, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * expansion_factor)
        self.swish = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * expansion_factor, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return residual + x


class ConvolutionModule(nn.Module):
    """
    Depthwise Separable Convolution Module.
    kernel_size=31 is typical for speech tasks (covers ~620ms at 20ms/frame).
    """
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd"
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.glu = GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        return residual + x


class BoundaryBiasedMultiHeadedAttention(nn.Module):
    """
    Multi-Head Self-Attention with Boundary Prior Injection.
    
    Core Innovation: Instead of hard gating, we inject boundary probability
    as a soft bias into the attention score matrix. This ensures that even
    when boundary prediction confidence drops in OOD scenarios, the attention
    mechanism remains functional and can still leverage local contrast.
    
    Mathematical Formulation:
        Attention(Q, K, V) = Softmax((QK^T / sqrt(d_k)) + α * BoundaryBias(p_bound)) V
    
    where BoundaryBias constructs a temporal contrast matrix that amplifies
    attention differences across predicted boundary frames.
    """
    def __init__(self, d_model, num_heads=8, dropout=0.1, boundary_bias_scale=1.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.boundary_bias_scale = boundary_bias_scale
        
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def _construct_boundary_bias(self, p_bound):
        """
        Construct attention bias matrix from boundary probabilities.
        
        Strategy: For each frame t with high boundary probability p_bound[t],
        we want to increase the attention contrast between frames before and
        after t. This is achieved by constructing a bias matrix that:
        1. Enhances attention within homogeneous regions (same side of boundary)
        2. Suppresses attention across boundary transitions
        
        Args:
            p_bound: [B, T, 1] boundary probability
        Returns:
            bias: [B, num_heads, T, T] attention bias matrix
        """
        B, T, _ = p_bound.shape
        p_bound = p_bound.squeeze(-1)
        
        boundary_strength = p_bound.unsqueeze(-1)
        boundary_diff = torch.abs(
            boundary_strength.unsqueeze(2) - boundary_strength.unsqueeze(1)
        )
        
        temporal_distance = torch.abs(
            torch.arange(T, device=p_bound.device).unsqueeze(0) - 
            torch.arange(T, device=p_bound.device).unsqueeze(1)
        ).float()
        temporal_distance = temporal_distance.unsqueeze(0).expand(B, -1, -1)
        
        local_mask = (temporal_distance <= 5).float()
        
        boundary_at_i = p_bound.unsqueeze(2).expand(-1, -1, T)
        boundary_at_j = p_bound.unsqueeze(1).expand(-1, T, -1)
        boundary_between = torch.maximum(boundary_at_i, boundary_at_j)
        
        bias = -boundary_between * boundary_diff * local_mask * self.boundary_bias_scale
        
        bias = bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        return bias

    def forward(self, x, p_bound_prior=None):
        """
        Args:
            x: [B, T, d_model] input features
            p_bound_prior: [B, T, 1] boundary probability (optional)
        Returns:
            output: [B, T, d_model]
            attn_weights: [B, num_heads, T, T] (for analysis)
        """
        residual = x
        x = self.norm(x)
        
        B, T, _ = x.shape
        
        Q = self.linear_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.linear_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.linear_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if p_bound_prior is not None:
            boundary_bias = self._construct_boundary_bias(p_bound_prior)
            scores = scores + boundary_bias
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        
        output = self.linear_out(context)
        output = self.dropout(output)
        
        return residual + output, attn_weights


class ConformerBlock(nn.Module):
    """
    Single Conformer Block with Boundary-Biased Attention.
    
    Architecture (Macaron-style):
        x = x + 0.5 * FFN(x)
        x = x + MHSA(x, boundary_prior)
        x = x + Conv(x)
        x = x + 0.5 * FFN(x)
        x = LayerNorm(x)
    """
    def __init__(self, d_model, num_heads=8, conv_kernel_size=31, 
                 expansion_factor=4, dropout=0.1, boundary_bias_scale=1.0):
        super().__init__()
        
        self.ff1 = FeedForwardModule(d_model, expansion_factor, dropout)
        self.mhsa = BoundaryBiasedMultiHeadedAttention(
            d_model, num_heads, dropout, boundary_bias_scale
        )
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ff2 = FeedForwardModule(d_model, expansion_factor, dropout)
        self.norm_final = nn.LayerNorm(d_model)

    def forward(self, x, p_bound_prior=None):
        """
        Args:
            x: [B, T, d_model]
            p_bound_prior: [B, T, 1] boundary probability
        Returns:
            x: [B, T, d_model]
            attn_weights: [B, num_heads, T, T]
        """
        x = self.ff1(x)
        x, attn_weights = self.mhsa(x, p_bound_prior)
        x = self.conv(x)
        x = self.ff2(x)
        x = self.norm_final(x)
        
        return x, attn_weights


class ConformerEncoder(nn.Module):
    """
    Stack of Conformer blocks for sequence modeling.
    Replaces Mamba mixer in the original BAT-Frame architecture.
    """
    def __init__(self, d_model, num_layers=6, num_heads=8, 
                 conv_kernel_size=31, expansion_factor=4, 
                 dropout=0.1, boundary_bias_scale=1.0):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([
            ConformerBlock(
                d_model, num_heads, conv_kernel_size, 
                expansion_factor, dropout, boundary_bias_scale
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, p_bound_prior=None):
        """
        Args:
            x: [B, T, d_model]
            p_bound_prior: [B, T, 1] boundary probability
        Returns:
            x: [B, T, d_model]
            all_attn_weights: list of [B, num_heads, T, T]
        """
        all_attn_weights = []
        
        for layer in self.layers:
            x, attn_weights = layer(x, p_bound_prior)
            all_attn_weights.append(attn_weights)
        
        return x, all_attn_weights
