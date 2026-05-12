#!/usr/bin/env python3
"""
RWKV-7 Agent Training with Cross-Chunk BPTT
============================================

End-to-end script for training a pretrained RWKV-7 .pth model on agent traces
(CodeAct, tool-use, GRPO, instruction-following) with:

- Checkpointed state-chain BPTT (gradients flow across chunk boundaries)
- Arbitrary token masking (only train on model-generated tokens)
- Score-weighted training (weight each trace by quality score)
- 4096-token chunks matching RWKV-7 pretrained ctx_len

Usage:
    python rwkv7_agent_train.py \
        --model /path/to/RWKV-x070-World-1.5B.pth \
        --data /path/to/traces.jsonl \
        --output ./checkpoints \
        --chunk_size 4096 \
        --lr 2e-5 \
        --epochs 3

Requirements:
    pip install torch deepspeed wandb
    + RWKV CUDA kernels (rwkv7_fast_fused) for production speed
"""

import os
import sys
import json
import math
import time
import random
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


# ============================================================================
# 1. MODEL LOADING — Load pretrained RWKV-7 .pth
# ============================================================================

def detect_model_config(state_dict: dict) -> dict:
    """
    Auto-detect model config from .pth state dict keys.
    
    RWKV .pth files don't store config separately — we infer it from
    the weight shapes. Works for all RWKV-7 World models.
    """
    # Count layers
    n_layer = 0
    for key in state_dict:
        if '.att.key.weight' in key:
            parts = key.split('.')
            layer_id = int(parts[1])
            n_layer = max(n_layer, layer_id + 1)
    
    # Get embedding dim from any weight
    n_embd = state_dict['blocks.0.att.key.weight'].shape[0]
    
    # Get vocab size from embeddings
    vocab_size = state_dict['emb.weight'].shape[0]
    
    # Get head size from time_decay shape
    # RWKV-7: time_decay is (n_embd,) or can infer head_size from att.r_k
    head_size = 64  # RWKV-7 default
    n_head = n_embd // head_size
    
    # Detect RWKV version from key patterns
    version = 'v7'  # default
    for key in state_dict:
        if 'att.a0' in key or 'att.v0' in key:
            version = 'v7'
            break
        if 'time_maa_x' in key:
            version = 'v6'
            break
    
    config = {
        'n_layer': n_layer,
        'n_embd': n_embd,
        'n_head': n_head,
        'head_size': head_size,
        'vocab_size': vocab_size,
        'version': version,
    }
    
    log.info(f"Detected model config: {json.dumps(config, indent=2)}")
    return config


class RWKV7TimeMixing(nn.Module):
    """
    RWKV-7 time-mixing block with state I/O for cross-chunk BPTT.
    
    For production: replace _wkv_scan with the rwkv7_fast_fused CUDA kernel.
    The CUDA kernel computes identical forward + backward (including gs)
    but runs 10-100x faster than this pure PyTorch scan.
    
    The pure PyTorch version is provided so the script works out-of-the-box
    without compiling CUDA kernels. Autograd computes gs automatically.
    """
    
    def __init__(self, layer_id, n_embd, n_head, head_size):
        super().__init__()
        self.layer_id = layer_id
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = head_size
        
        # These will be loaded from the pretrained .pth
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)
        self.gate = nn.Linear(n_embd, n_embd, bias=False)
        self.ln_x = nn.GroupNorm(n_head, n_embd, eps=64e-5)
        
        # RWKV-7 specific parameters (loaded from .pth)
        self.time_decay = nn.Parameter(torch.zeros(n_embd))
        self.time_first = nn.Parameter(torch.zeros(n_head, head_size))
        
        # Token shift mixing factors
        self.time_maa_r = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.time_maa_k = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.time_maa_v = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.time_maa_w = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.time_maa_g = nn.Parameter(torch.zeros(1, 1, n_embd))
    
    def forward(self, x, state):
        """
        Args:
            x: (B, T, C) hidden states
            state: (wkv_state, shift_state)
                wkv_state: (B, n_head, head_size, head_size) — NOT detached
                shift_state: (B, 1, n_embd) — last token from prev chunk
        Returns:
            (output, (new_wkv_state, new_shift_state))
        """
        B, T, C = x.shape
        H = self.n_head
        S = self.head_size
        
        wkv_state, shift_state = state
        
        # Token shift mixing
        xx = torch.cat([shift_state, x[:, :-1, :]], dim=1)
        new_shift_state = x[:, -1:, :].clone()
        
        xr = x + (xx - x) * self.time_maa_r
        xk = x + (xx - x) * self.time_maa_k
        xv = x + (xx - x) * self.time_maa_v
        xw = x + (xx - x) * self.time_maa_w
        xg = x + (xx - x) * self.time_maa_g
        
        r = self.receptance(xr).view(B, T, H, S)
        k = self.key(xk).view(B, T, H, S)
        v = self.value(xv).view(B, T, H, S)
        g = torch.sigmoid(self.gate(xg))
        
        # Decay
        w = (-torch.exp(self.time_decay.float())).view(H, S)
        w = torch.exp(w)  # (H, S) in (0, 1)
        
        # WKV scan — THIS IS THE CRITICAL PART
        y, new_wkv_state = self._wkv_scan(r, k, v, w, wkv_state)
        
        y = y.reshape(B * T, C)
        y = self.ln_x(y).view(B, T, C) * g
        y = self.output(y)
        
        return y, (new_wkv_state, new_shift_state)
    
    def _wkv_scan(self, r, k, v, w, state_in):
        """
        Pure PyTorch WKV scan with state I/O.
        
        PRODUCTION: Replace this with:
            y, state_out = RUN_CUDA_RWKV7_STATE(r, k, v, w, u, state_in)
        
        The CUDA kernel (rwkv7_fast_fused state-passing variant) computes
        identical results but includes an optimized backward that produces gs.
        
        With pure PyTorch, autograd handles gs automatically.
        """
        B, T, H, S = r.shape
        s = state_in.float()
        
        outputs = []
        for t in range(T):
            s = s * w.unsqueeze(0).unsqueeze(-1) + \
                k[:, t].unsqueeze(-1) * v[:, t].unsqueeze(-2)
            y_t = torch.einsum('bhs,bhsd->bhd', r[:, t].float(), s)
            outputs.append(y_t.to(r.dtype))
        
        return torch.stack(outputs, dim=1), s.to(r.dtype)
    
    def init_state(self, batch_size, device, dtype=torch.bfloat16):
        H, S = self.n_head, self.head_size
        wkv = torch.zeros(batch_size, H, S, S, device=device, dtype=dtype)
        shift = torch.zeros(batch_size, 1, self.n_embd, device=device, dtype=dtype)
        return (wkv, shift)


class RWKV7ChannelMixing(nn.Module):
    """RWKV-7 channel mixing (feed-forward) block."""
    
    def __init__(self, layer_id, n_embd):
        super().__init__()
        self.layer_id = layer_id
        self.n_embd = n_embd
        
        self.key = nn.Linear(n_embd, n_embd * 4, bias=False)
        self.value = nn.Linear(n_embd * 4, n_embd, bias=False)
        
        self.time_maa_k = nn.Parameter(torch.zeros(1, 1, n_embd))
    
    def forward(self, x, shift_state):
        B, T, C = x.shape
        xx = torch.cat([shift_state, x[:, :-1, :]], dim=1)
        new_shift = x[:, -1:, :].clone()
        
        xk = x + (xx - x) * self.time_maa_k
        k = self.key(xk)
        k = torch.relu(k) ** 2
        return self.value(k), new_shift
    
    def init_state(self, batch_size, device, dtype=torch.bfloat16):
        return torch.zeros(batch_size, 1, self.n_embd, device=device, dtype=dtype)


class RWKV7Block(nn.Module):
    """Single RWKV-7 block (LayerNorm + TimeMix + LayerNorm + ChannelMix)."""
    
    def __init__(self, layer_id, n_embd, n_head, head_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.att = RWKV7TimeMixing(layer_id, n_embd, n_head, head_size)
        self.ffn = RWKV7ChannelMixing(layer_id, n_embd)
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)
    
    def forward(self, x, state):
        att_state, ffn_shift = state
        dx, new_att_state = self.att(self.ln1(x), att_state)
        x = x + dx
        dx, new_ffn_shift = self.ffn(self.ln2(x), ffn_shift)
        x = x + dx
        return x, (new_att_state, new_ffn_shift)
    
    def init_state(self, batch_size, device, dtype):
        return (
            self.att.init_state(batch_size, device, dtype),
            self.ffn.init_state(batch_size, device, dtype)
        )


class RWKV7Model(nn.Module):
    """
    Full RWKV-7 model with state I/O for cross-chunk BPTT.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        n_embd = config['n_embd']
        n_layer = config['n_layer']
        n_head = config['n_head']
        head_size = config['head_size']
        vocab_size = config['vocab_size']
        
        self.emb = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([
            RWKV7Block(i, n_embd, n_head, head_size)
            for i in range(n_layer)
        ])
        self.ln_out = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
    
    def forward(self, idx, state=None):
        """
        Args:
            idx: (B, T) token ids
            state: list of block states, or None
        Returns:
            logits: (B, T, V)
            new_state: list of block states (IN THE GRAPH — not detached)
        """
        B, T = idx.shape
        device = idx.device
        dtype = self.emb.weight.dtype
        
        if state is None:
            state = [blk.init_state(B, device, dtype) for blk in self.blocks]
        
        x = self.emb(idx)
        if hasattr(self.blocks[0], 'ln0'):
            x = self.blocks[0].ln0(x)
        
        new_state = []
        for i, block in enumerate(self.blocks):
            x, blk_state = block(x, state[i])
            new_state.append(blk_state)
        
        logits = self.head(self.ln_out(x))
        return logits, new_state
    
    def flatten_state(self, state):
        """Convert nested state structure to flat tensor list for checkpointing."""
        flat = []
        for (att_state, ffn_shift) in state:
            wkv, att_shift = att_state
            flat.extend([wkv, att_shift, ffn_shift])
        return flat
    
    def unflatten_state(self, flat):
        """Convert flat tensor list back to nested state structure."""
        state = []
        for i in range(0, len(flat), 3):
            att_state = (flat[i], flat[i+1])
            ffn_shift = flat[i+2]
            state.append((att_state, ffn_shift))
        return state
    
    def forward_for_checkpoint(self, idx, *flat_state):
        """Wrapper for torch.utils.checkpoint (needs flat tensor args)."""
        state = self.unflatten_state(list(flat_state))
        logits, new_state = self.forward(idx, state)
        return (logits, *self.flatten_state(new_state))

    @classmethod
    def from_pretrained(cls, path: str, device='cpu'):
        """Load from RWKV .pth file."""
        log.info(f"Loading model from {path}")
        sd = torch.load(path, map_location=device, weights_only=True)
        config = detect_model_config(sd)
        model = cls(config)
        
        # Map weights (RWKV .pth uses different naming than our module)
        # This is a simplified mapping — adjust for your specific .pth format
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            log.warning(f"Missing keys: {len(missing)} (expected for arch differences)")
            for k in missing[:10]:
                log.warning(f"  {k}")
        if unexpected:
            log.warning(f"Unexpected keys: {len(unexpected)}")
            for k in unexpected[:10]:
                log.warning(f"  {k}")
        
        log.info(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
        return model, config


# ============================================================================
# 2. DATA — Agent trace loading with masking and scoring
# ============================================================================

@dataclass
class AgentTrace:
    """A single agent trace with per-token mask and optional score."""
    token_ids: list[int]
    loss_mask: list[int]    # 1 = train, 0 = skip
    score: float = 1.0      # quality weight for this trace
    metadata: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.token_ids)


class TraceTokenizer:
    """
    Converts structured agent traces to token_ids + loss_mask.
    
    Supported input formats:
    
    Format 1 — Role-tagged turns (recommended for CodeAct):
    {
        "score": 0.85,
        "turns": [
            {"role": "system", "content": "You are..."},
            {"role": "user", "content": "Write a function..."},
            {"role": "assistant", "content": "<think>I need to...</think>"},
            {"role": "action", "content": "```python\ndef solve(): ...```"},
            {"role": "observation", "content": "Output: [1, 2, 3]"},
            {"role": "assistant", "content": "Here is the solution..."}
        ]
    }
    
    Format 2 — Pre-tokenized with mask:
    {
        "score": 0.85,
        "tokens": [1, 504, 23, ...],
        "mask": [0, 0, 0, 1, 1, ...]
    }
    
    Format 3 — Raw text with role markers:
    {
        "score": 0.85,
        "text": "User: ...\n\nAssistant: ...\n\n",
        "train_spans": [[100, 250], [300, 450]]  // char ranges to train on
    }
    """
    
    # Roles where loss is computed
    TRAIN_ROLES = {'assistant', 'thought', 'action', 'answer', 'model'}
    MASK_ROLES = {'system', 'user', 'observation', 'tool_result', 'environment', 'human'}
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def process(self, entry: dict) -> AgentTrace:
        """Convert a single data entry to an AgentTrace."""
        score = entry.get('score', 1.0)
        
        if 'tokens' in entry and 'mask' in entry:
            return self._from_pretokenized(entry, score)
        elif 'turns' in entry:
            return self._from_turns(entry, score)
        elif 'text' in entry:
            return self._from_text(entry, score)
        else:
            raise ValueError(f"Unknown data format. Keys: {list(entry.keys())}")
    
    def _from_pretokenized(self, entry, score):
        return AgentTrace(
            token_ids=entry['tokens'],
            loss_mask=entry['mask'],
            score=score,
        )
    
    def _from_turns(self, entry, score):
        all_tokens = []
        all_mask = []
        
        for turn in entry['turns']:
            role = turn['role']
            content = turn['content']
            trainable = role in self.TRAIN_ROLES
            
            # Encode with role markers
            prefix = f"<|{role}|>\n"
            suffix = f"\n<|end_{role}|>\n"
            
            prefix_tok = self._encode(prefix)
            content_tok = self._encode(content)
            suffix_tok = self._encode(suffix)
            
            all_tokens.extend(prefix_tok + content_tok + suffix_tok)
            all_mask.extend(
                [0] * len(prefix_tok) +
                [1 if trainable else 0] * len(content_tok) +
                [0] * len(suffix_tok)
            )
        
        # Shift mask to align with TARGETS (predict next token)
        target_mask = all_mask[1:] + [0]
        
        return AgentTrace(
            token_ids=all_tokens,
            loss_mask=target_mask,
            score=score,
        )
    
    def _from_text(self, entry, score):
        text = entry['text']
        tokens = self._encode(text)
        mask = [0] * len(tokens)
        
        if 'train_spans' in entry:
            char_to_tok = self._build_char_to_token_map(text, tokens)
            for start_char, end_char in entry['train_spans']:
                start_tok = char_to_tok.get(start_char, 0)
                end_tok = char_to_tok.get(end_char, len(tokens))
                for i in range(start_tok, min(end_tok, len(tokens))):
                    mask[i] = 1
        
        target_mask = mask[1:] + [0]
        return AgentTrace(token_ids=tokens, loss_mask=target_mask, score=score)
    
    def _encode(self, text: str) -> list[int]:
        if hasattr(self.tokenizer, 'encode'):
            return self.tokenizer.encode(text)
        return list(text.encode('utf-8'))
    
    def _build_char_to_token_map(self, text, tokens):
        # Simplified — in production, use tokenizer's offset mapping
        ratio = len(tokens) / max(len(text), 1)
        return {i: int(i * ratio) for i in range(len(text))}


class AgentTraceDataset:
    """
    Loads agent traces from JSONL with score-weighted sampling.
    """
    
    def __init__(self, path: str, tokenizer, max_traces=None):
        self.traces = []
        processor = TraceTokenizer(tokenizer)
        
        with open(path) as f:
            for i, line in enumerate(f):
                if max_traces and i >= max_traces:
                    break
                entry = json.loads(line.strip())
                trace = processor.process(entry)
                self.traces.append(trace)
        
        log.info(f"Loaded {len(self.traces)} traces from {path}")
        
        # Stats
        total_tokens = sum(len(t) for t in self.traces)
        trained_tokens = sum(sum(t.loss_mask) for t in self.traces)
        log.info(f"  Total tokens: {total_tokens:,}")
        log.info(f"  Trained tokens: {trained_tokens:,} ({trained_tokens/total_tokens:.1%})")
        scores = [t.score for t in self.traces]
        log.info(f"  Score range: [{min(scores):.2f}, {max(scores):.2f}]")
    
    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, idx):
        return self.traces[idx]
    
    def score_weighted_order(self, epoch_seed=0):
        """
        Yield trace indices weighted by score.
        
        Higher-scored traces appear more frequently.
        This implements the score-weighting approach:
        instead of multiplying loss by score (which changes gradient magnitude),
        we sample traces proportional to score (which changes frequency).
        
        Effect: a trace with score 2.0 is seen ~2x as often as score 1.0.
        The gradients per example are unmodified — only the sampling changes.
        """
        rng = random.Random(epoch_seed)
        
        weights = [max(t.score, 0.01) for t in self.traces]
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        
        # Generate an epoch's worth of samples
        n_samples = len(self.traces)
        indices = rng.choices(range(len(self.traces)), weights=probs, k=n_samples)
        rng.shuffle(indices)
        
        for idx in indices:
            yield idx


# ============================================================================
# 3. CORE ENGINE — Cross-chunk BPTT with activation checkpointing
# ============================================================================

class CrossChunkTrainer:
    """
    The training engine. Implements checkpointed state-chain BPTT.
    """
    
    def __init__(
        self,
        model: RWKV7Model,
        optimizer: torch.optim.Optimizer,
        chunk_size: int = 4096,
        grad_clip: float = 1.0,
        grad_accum_steps: int = 1,
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
        use_cross_chunk_bptt: bool = True,
        use_checkpointing: bool = True,
        score_mode: str = 'sample',   # 'sample' or 'weight'
    ):
        self.model = model
        self.optimizer = optimizer
        self.chunk_size = chunk_size
        self.grad_clip = grad_clip
        self.grad_accum_steps = grad_accum_steps
        self.device = device
        self.dtype = dtype
        self.use_cross_chunk_bptt = use_cross_chunk_bptt
        self.use_checkpointing = use_checkpointing
        self.score_mode = score_mode
        self.step = 0
    
    def train_trace(self, trace: AgentTrace) -> dict:
        """
        Train on a single agent trace with cross-chunk BPTT.
        
        Returns stats dict.
        """
        tokens = torch.tensor([trace.token_ids], dtype=torch.long, device=self.device)
        targets = torch.tensor([trace.token_ids[1:] + [0]], dtype=torch.long, device=self.device)
        mask = torch.tensor([trace.loss_mask], dtype=self.dtype, device=self.device)
        
        B, total_len = tokens.shape
        n_chunks = math.ceil(total_len / self.chunk_size)
        
        # Initialize state
        state = [blk.init_state(B, self.device, self.dtype) for blk in self.model.blocks]
        
        chunk_losses = []
        total_trained = 0
        total_masked = 0
        
        with torch.amp.autocast('cuda', dtype=self.dtype, enabled=self.dtype != torch.float32):
            for c in range(n_chunks):
                start = c * self.chunk_size
                end = min(start + self.chunk_size, total_len)
                
                chunk_tok = tokens[:, start:end]
                chunk_tgt = targets[:, start:end]
                chunk_mask = mask[:, start:end]
                
                # Forward with optional checkpointing
                if self.use_checkpointing and c > 0 and self.use_cross_chunk_bptt:
                    flat = self.model.flatten_state(state)
                    outputs = checkpoint(
                        self.model.forward_for_checkpoint,
                        chunk_tok,
                        *flat,
                        use_reentrant=False,
                    )
                    logits = outputs[0]
                    new_state = self.model.unflatten_state(list(outputs[1:]))
                else:
                    logits, new_state = self.model.forward(chunk_tok, state)
                
                # State propagation — THE critical choice
                if self.use_cross_chunk_bptt:
                    state = new_state           # gradients flow through
                else:
                    state = self._detach_state(new_state)  # truncated (baseline)
                
                # Masked loss
                n_trained = chunk_mask.sum().item()
                n_masked = (end - start) - n_trained
                total_trained += n_trained
                total_masked += n_masked
                
                if n_trained > 0:
                    loss_per_tok = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        chunk_tgt.view(-1),
                        reduction='none'
                    ).view(B, -1)
                    chunk_loss = (loss_per_tok * chunk_mask).sum()
                    chunk_losses.append(chunk_loss)
        
        # Total loss
        if total_trained > 0:
            total_loss = sum(chunk_losses) / total_trained
            
            # Score weighting (if using 'weight' mode)
            if self.score_mode == 'weight':
                total_loss = total_loss * trace.score
            
            # Scale for gradient accumulation
            scaled_loss = total_loss / self.grad_accum_steps
            scaled_loss.backward()
        else:
            total_loss = torch.tensor(0.0, device=self.device)
        
        # Optimizer step
        self.step += 1
        did_step = False
        if self.step % self.grad_accum_steps == 0:
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            did_step = True
        
        return {
            'loss': total_loss.item(),
            'trained_tokens': int(total_trained),
            'masked_tokens': int(total_masked),
            'n_chunks': n_chunks,
            'trace_len': len(trace),
            'score': trace.score,
            'did_optimizer_step': did_step,
        }
    
    def _detach_state(self, state):
        """Detach all state tensors (for baseline/truncated mode)."""
        result = []
        for (att_state, ffn_shift) in state:
            wkv, att_shift = att_state
            result.append((
                (wkv.detach(), att_shift.detach()),
                ffn_shift.detach()
            ))
        return result


# ============================================================================
# 4. OPTIMIZER — RWKV-specific parameter groups
# ============================================================================

def build_optimizer(model, lr, lr_final, weight_decay=0.1, betas=(0.9, 0.99)):
    """
    Build AdamW optimizer with RWKV-specific parameter grouping.
    
    IMPORTANT (from BlinkDL):
    - Only apply weight decay to large matrix parameters (projections)
    - NOT to LayerNorm, NOT to small vectors (time_decay, time_mix, etc.)
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Weight decay only on large 2D matrices
        if param.dim() >= 2 and min(param.shape) >= 128:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    
    log.info(f"Optimizer: {len(decay_params)} decay params, {len(no_decay_params)} no-decay params")
    
    return torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=lr, betas=betas, eps=1e-8)


# ============================================================================
# 5. MAIN TRAINING LOOP
# ============================================================================

def train(args):
    """Main training function."""
    
    # ---- Setup ----
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float32
    
    log.info(f"Device: {device}, Precision: {args.precision}")
    log.info(f"Cross-chunk BPTT: {args.cross_chunk_bptt}")
    log.info(f"Score mode: {args.score_mode}")
    log.info(f"Chunk size: {args.chunk_size}")
    
    # ---- Load model ----
    model, config = RWKV7Model.from_pretrained(args.model, device='cpu')
    model = model.to(device=device, dtype=dtype)
    model.train()
    
    # ---- Load data ----
    # Use a simple tokenizer wrapper (replace with RWKV World tokenizer)
    class SimpleTokenizer:
        def __init__(self):
            try:
                from rwkv.utils import PIPELINE
                from rwkv.model import RWKV as RWKVInference
                self._pipeline = PIPELINE(None, "rwkv_vocab_v20230424")
                self.encode = self._pipeline.encode
                self.decode = self._pipeline.decode
            except ImportError:
                log.warning("RWKV tokenizer not found, using UTF-8 byte fallback")
                self.encode = lambda text: list(text.encode('utf-8'))
                self.decode = lambda ids: bytes(ids).decode('utf-8', errors='replace')
    
    tokenizer = SimpleTokenizer()
    dataset = AgentTraceDataset(args.data, tokenizer, max_traces=args.max_traces)
    
    # ---- Optimizer ----
    optimizer = build_optimizer(
        model, lr=args.lr, lr_final=args.lr_final,
        weight_decay=args.weight_decay
    )
    
    # ---- Trainer ----
    trainer = CrossChunkTrainer(
        model=model,
        optimizer=optimizer,
        chunk_size=args.chunk_size,
        grad_clip=args.grad_clip,
        grad_accum_steps=args.grad_accum,
        device=device,
        dtype=dtype,
        use_cross_chunk_bptt=args.cross_chunk_bptt,
        use_checkpointing=args.checkpoint,
        score_mode=args.score_mode,
    )
    
    # ---- Wandb ----
    if args.wandb:
        try:
            import wandb
            wandb.init(project=args.wandb, config=vars(args))
        except ImportError:
            log.warning("wandb not installed, skipping")
            args.wandb = None
    
    # ---- Training ----
    os.makedirs(args.output, exist_ok=True)
    
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_start = time.time()
        
        # Score-weighted sampling
        if args.score_mode == 'sample':
            trace_order = list(dataset.score_weighted_order(epoch_seed=epoch))
        else:
            trace_order = list(range(len(dataset)))
            random.Random(epoch).shuffle(trace_order)
        
        for trace_idx_in_epoch, data_idx in enumerate(trace_order):
            trace = dataset[data_idx]
            
            stats = trainer.train_trace(trace)
            global_step += 1
            
            epoch_loss += stats['loss'] * stats['trained_tokens']
            epoch_tokens += stats['trained_tokens']
            
            # Logging
            if global_step % args.log_every == 0:
                avg_loss = epoch_loss / max(epoch_tokens, 1)
                elapsed = time.time() - epoch_start
                tps = epoch_tokens / max(elapsed, 1)
                
                log.info(
                    f"Epoch {epoch+1}/{args.epochs} | "
                    f"Step {global_step} | "
                    f"Trace {trace_idx_in_epoch+1}/{len(trace_order)} | "
                    f"Loss: {stats['loss']:.4f} | "
                    f"Avg: {avg_loss:.4f} | "
                    f"Chunks: {stats['n_chunks']} | "
                    f"Score: {stats['score']:.2f} | "
                    f"Tok/s: {tps:.0f}"
                )
                
                if args.wandb:
                    wandb.log({
                        'loss': stats['loss'],
                        'avg_loss': avg_loss,
                        'trained_tokens': stats['trained_tokens'],
                        'score': stats['score'],
                        'n_chunks': stats['n_chunks'],
                        'tokens_per_sec': tps,
                    }, step=global_step)
            
            # Save checkpoint
            if global_step % args.save_every == 0:
                path = os.path.join(args.output, f'rwkv7-agent-step{global_step}.pth')
                torch.save(model.state_dict(), path)
                log.info(f"Saved checkpoint: {path}")
        
        # Epoch summary
        avg_epoch_loss = epoch_loss / max(epoch_tokens, 1)
        elapsed = time.time() - epoch_start
        log.info(
            f"=== Epoch {epoch+1} complete | "
            f"Loss: {avg_epoch_loss:.4f} | "
            f"Tokens: {epoch_tokens:,} | "
            f"Time: {elapsed:.0f}s ==="
        )
        
        # Save epoch checkpoint
        path = os.path.join(args.output, f'rwkv7-agent-epoch{epoch+1}.pth')
        torch.save(model.state_dict(), path)
        log.info(f"Saved epoch checkpoint: {path}")
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            path = os.path.join(args.output, 'rwkv7-agent-best.pth')
            torch.save(model.state_dict(), path)
            log.info(f"New best model: {path} (loss={best_loss:.4f})")
    
    # Final save
    path = os.path.join(args.output, 'rwkv7-agent-final.pth')
    torch.save(model.state_dict(), path)
    log.info(f"Training complete. Final model: {path}")


# ============================================================================
# 6. DATA GENERATION UTILITIES
# ============================================================================

def generate_example_data(output_path: str):
    """Generate example training data in all supported formats."""
    
    examples = [
        # Format 1: Role-tagged turns (CodeAct style)
        {
            "score": 1.0,
            "turns": [
                {"role": "system", "content": "You are a coding assistant. Use <think> for reasoning and <action> for code execution."},
                {"role": "user", "content": "Write a function to check if a number is prime."},
                {"role": "assistant", "content": "<think>I'll write an efficient primality test checking divisibility up to sqrt(n).</think>"},
                {"role": "action", "content": "```python\ndef is_prime(n):\n    if n < 2: return False\n    if n < 4: return True\n    if n % 2 == 0 or n % 3 == 0: return False\n    i = 5\n    while i * i <= n:\n        if n % i == 0 or n % (i+2) == 0: return False\n        i += 6\n    return True\n\nprint([n for n in range(20) if is_prime(n)])\n```"},
                {"role": "observation", "content": "[2, 3, 5, 7, 11, 13, 17, 19]"},
                {"role": "assistant", "content": "Here's the function. It works by checking divisibility up to the square root of n, skipping even numbers and multiples of 3 for efficiency."},
            ]
        },
        # High-scoring trace (will be sampled more often)
        {
            "score": 2.0,
            "turns": [
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": "Sort a list of dicts by a key."},
                {"role": "assistant", "content": "<think>Use sorted() with a key function.</think>"},
                {"role": "action", "content": "```python\ndata = [{'name': 'Bob', 'age': 30}, {'name': 'Alice', 'age': 25}]\nresult = sorted(data, key=lambda x: x['age'])\nprint(result)\n```"},
                {"role": "observation", "content": "[{'name': 'Alice', 'age': 25}, {'name': 'Bob', 'age': 30}]"},
                {"role": "assistant", "content": "Use `sorted()` with `key=lambda x: x['age']` to sort by any dict key."},
            ]
        },
        # Low-scoring trace (sampled less often)
        {
            "score": 0.3,
            "turns": [
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": "Reverse a string."},
                {"role": "assistant", "content": "You can reverse a string using slicing: `s[::-1]`."},
            ]
        },
        # Format 2: Pre-tokenized
        {
            "score": 1.0,
            "tokens": list(range(100, 200)),
            "mask": [0]*30 + [1]*40 + [0]*20 + [1]*10,
        },
    ]
    
    with open(output_path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')
    
    log.info(f"Generated {len(examples)} example traces in {output_path}")
    return output_path


# ============================================================================
# 7. CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='RWKV-7 Agent Training with Cross-Chunk BPTT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python rwkv7_agent_train.py \\
      --model RWKV-x070-World-1.5B.pth \\
      --data traces.jsonl \\
      --output ./checkpoints

  # Full BPTT with score-weighted sampling
  python rwkv7_agent_train.py \\
      --model RWKV-x070-World-1.5B.pth \\
      --data traces.jsonl \\
      --cross_chunk_bptt \\
      --score_mode sample \\
      --lr 2e-5 \\
      --epochs 5

  # Generate example training data
  python rwkv7_agent_train.py --generate_data example_traces.jsonl

  # Baseline comparison (truncated BPTT)
  python rwkv7_agent_train.py \\
      --model model.pth --data traces.jsonl \\
      --no_cross_chunk_bptt
        """
    )
    
    # Model
    parser.add_argument('--model', type=str, help='Path to pretrained RWKV-7 .pth file')
    parser.add_argument('--output', type=str, default='./checkpoints', help='Output directory')
    
    # Data
    parser.add_argument('--data', type=str, help='Path to training data (JSONL)')
    parser.add_argument('--max_traces', type=int, default=None, help='Max traces to load')
    parser.add_argument('--generate_data', type=str, default=None,
                        help='Generate example data to this path (then exit)')
    
    # Training
    parser.add_argument('--chunk_size', type=int, default=4096, help='Chunk size (match pretrain ctx_len)')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--lr_final', type=float, default=2e-6, help='Final learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--grad_accum', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--precision', type=str, default='bf16', choices=['bf16', 'fp32'])
    
    # Cross-chunk BPTT
    parser.add_argument('--cross_chunk_bptt', action='store_true', default=True,
                        help='Enable cross-chunk BPTT (default: enabled)')
    parser.add_argument('--no_cross_chunk_bptt', action='store_true',
                        help='Disable cross-chunk BPTT (use truncated baseline)')
    parser.add_argument('--checkpoint', action='store_true', default=True,
                        help='Use activation checkpointing (recommended)')
    parser.add_argument('--no_checkpoint', action='store_true',
                        help='Disable activation checkpointing')
    
    # Score handling
    parser.add_argument('--score_mode', type=str, default='sample',
                        choices=['sample', 'weight', 'none'],
                        help='How to use trace scores: '
                             'sample=weighted sampling frequency, '
                             'weight=multiply loss by score, '
                             'none=ignore scores')
    
    # Logging
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--wandb', type=str, default=None, help='Wandb project name')
    
    args = parser.parse_args()
    
    # Handle flag overrides
    if args.no_cross_chunk_bptt:
        args.cross_chunk_bptt = False
    if args.no_checkpoint:
        args.checkpoint = False
    
    # Generate data mode
    if args.generate_data:
        generate_example_data(args.generate_data)
        return
    
    # Validate required args
    if not args.model or not args.data:
        parser.error("--model and --data are required for training")
    
    train(args)


if __name__ == '__main__':
    main()
