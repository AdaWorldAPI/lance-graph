# ═══════════════════════════════════════════════════════════════
# REFERENCE IMPLEMENTATION — NOT THE PRODUCTION PATH
#
# The canonical Rust equivalent is:
#   cargo run --release --example tts_full_inference \
#     --manifest-path crates/thinking-engine/Cargo.toml
#
# That example runs the full 33-layer transformer + 128-step
# autoregressive + conv decoder → 24kHz WAV in pure Rust with
# AVX-512 F32x16 FMA + AMX polyfill. No Python runtime needed.
#
# This script exists for:
#   - Cross-checking Rust output against HuggingFace reference
#   - Quick prototyping before porting to Rust
#   - HF model download (huggingface_hub auth flow)
#
# See also: docs/CODEC_INVARIANTS_AND_EXPERIMENTS.md
# ═══════════════════════════════════════════════════════════════
#!/usr/bin/env python3
"""Qwen3-TTS-12Hz-0.6B full inference: text → speech WAV.

All weights loaded from safetensors. No HuggingFace transformers library.
33 transformer layers (28 talker + 5 code predictor) + conv decoder.
"""
import sys, os, time, torch, soundfile as sf
from safetensors.torch import load_file

MODEL = '/home/user/models/qwen3-tts-0.6b/model.safetensors'
TOKENIZER = '/home/user/models/qwen3-tts-0.6b/speech_tokenizer/model.safetensors'
OUT_DIR = '/home/user/lance-graph/data/tts-cascade'
EPS = 1e-6

# --- Primitives ---
def rms_norm(x, w): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w
def silu(x): return x * torch.sigmoid(x)
def gelu(x): return x * 0.5 * (1.0 + torch.erf(x / 1.4142135623730951))
def snake(x, alpha): return x + (1.0 / alpha.abs().clamp(min=1e-4)) * torch.sin(alpha * x).pow(2)

def rope(x, n):
    d = x.shape[-1]; pos = torch.arange(n).unsqueeze(1).float()
    f = 1.0 / (10000.0 ** (torch.arange(0, d, 2).float() / d))
    a = pos * f; o = torch.zeros_like(x)
    o[..., ::2] = x[..., ::2]*a.cos() - x[..., 1::2]*a.sin()
    o[..., 1::2] = x[..., ::2]*a.sin() + x[..., 1::2]*a.cos()
    return o

def qwen3_layer(h, pfx, w, n_heads, n_kv_heads, seq):
    """One Qwen3 transformer layer."""
    q_dim = w[f'{pfx}.self_attn.q_proj.weight'].shape[0]
    k_dim = w[f'{pfx}.self_attn.k_proj.weight'].shape[0]
    hd = q_dim // n_heads
    n_kv = k_dim // hd

    hn = rms_norm(h, w[f'{pfx}.input_layernorm.weight'])
    q = (hn @ w[f'{pfx}.self_attn.q_proj.weight'].T).view(seq, n_heads, hd)
    k = (hn @ w[f'{pfx}.self_attn.k_proj.weight'].T).view(seq, n_kv, hd)
    v = (hn @ w[f'{pfx}.self_attn.v_proj.weight'].T).view(seq, n_kv, hd)
    if f'{pfx}.self_attn.q_norm.weight' in w:
        q = rms_norm(q, w[f'{pfx}.self_attn.q_norm.weight'])
        k = rms_norm(k, w[f'{pfx}.self_attn.k_norm.weight'])
    for j in range(n_heads): q[:,j] = rope(q[:,j], seq)
    for j in range(n_kv): k[:,j] = rope(k[:,j], seq)
    k = k.repeat_interleave(n_heads // n_kv, dim=1)
    v = v.repeat_interleave(n_heads // n_kv, dim=1)
    sc = torch.einsum('shd,thd->hst', q, k) / (hd ** 0.5)
    sc = sc + torch.triu(torch.full((seq, seq), float('-inf')), 1)
    out = torch.einsum('hst,thd->shd', sc.softmax(-1), v).reshape(seq, -1)
    h = h + out @ w[f'{pfx}.self_attn.o_proj.weight'].T

    hn = rms_norm(h, w[f'{pfx}.post_attention_layernorm.weight'])
    h = h + (silu(hn @ w[f'{pfx}.mlp.gate_proj.weight'].T) * (hn @ w[f'{pfx}.mlp.up_proj.weight'].T)) @ w[f'{pfx}.mlp.down_proj.weight'].T
    return h

def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "Hello, world. This is a test of text to speech."
    print(f'Text: "{text}"')
    os.makedirs(OUT_DIR, exist_ok=True)

    t0 = time.time()
    print('Loading model (1.8 GB)...')
    w = {k: v.float() for k, v in load_file(MODEL, device='cpu').items()}
    print(f'  {len(w)} tensors in {time.time()-t0:.1f}s')

    t1 = time.time()
    print('Loading speech tokenizer (651 MB)...')
    tw = {k: v.float() for k, v in load_file(TOKENIZER, device='cpu').items()}
    print(f'  {len(tw)} tensors in {time.time()-t1:.1f}s')

    # === TALKER (28 layers) ===
    tokens = [151672] + [ord(c) for c in text] + [151673]
    seq = len(tokens)
    print(f'\n=== TALKER ({seq} tokens, 28 layers) ===')

    x = w['talker.model.text_embedding.weight'][tokens]
    x = silu(x @ w['talker.text_projection.linear_fc1.weight'].T + w['talker.text_projection.linear_fc1.bias'])
    x = x @ w['talker.text_projection.linear_fc2.weight'].T + w['talker.text_projection.linear_fc2.bias']
    print(f'  Embedded+projected: {x.shape} RMS={x.norm(dim=-1).mean():.3f}')

    t2 = time.time()
    for i in range(28):
        x = qwen3_layer(x, f'talker.model.layers.{i}', w, 16, 8, seq)
        if i % 7 == 0: print(f'  Layer {i:2d}: RMS={x.norm(dim=-1).mean():.3f}')
    print(f'  28 layers in {time.time()-t2:.1f}s')

    x = rms_norm(x, w['talker.model.norm.weight'])
    ct = (x @ w['talker.codec_head.weight'].T).argmax(dim=-1)
    print(f'  Codec tokens: {ct[:8].tolist()}...')

    # === CODE PREDICTOR (5 layers) ===
    print(f'\n=== CODE PREDICTOR (5 layers) ===')
    h = w['talker.model.codec_embedding.weight'][ct]
    for i in range(5):
        h = qwen3_layer(h, f'talker.code_predictor.model.layers.{i}', w, 16, 8, seq)
    h = rms_norm(h, w['talker.code_predictor.model.norm.weight'])
    print(f'  CP output: RMS={h.norm(dim=-1).mean():.3f}')

    # 15 lm_heads → codec tokens (0-2047)
    codes = torch.stack([(h @ w[f'talker.code_predictor.lm_head.{g}.weight'].T).argmax(-1) for g in range(15)], dim=-1)
    print(f'  Audio codes: {codes.shape}')
    print(f'  Frame 0: {codes[0,:5].tolist()}...')
    print(f'  Frame {seq//2}: {codes[seq//2,:5].tolist()}...')

    # === SPEECH TOKENIZER DECODER ===
    print(f'\n=== DECODER (8 pre-transformer + conv stack) ===')

    def get_cb(pfx):
        es = tw[f'{pfx}._codebook.embedding_sum']
        cu = tw[f'{pfx}._codebook.cluster_usage']
        return es / cu.unsqueeze(1).clamp(min=1.0)

    cb0 = get_cb('decoder.quantizer.rvq_first.vq.layers.0')
    cbs = [get_cb(f'decoder.quantizer.rvq_rest.vq.layers.{i}') for i in range(15)]

    e0 = cb0[codes[:, 0]]
    p0 = torch.nn.functional.conv1d(e0.T.unsqueeze(0), tw['decoder.quantizer.rvq_first.output_proj.weight']).squeeze(0).T
    er = sum(cbs[i][codes[:, i+1]] for i in range(min(14, codes.shape[1]-1)))
    pr = torch.nn.functional.conv1d(er.T.unsqueeze(0), tw['decoder.quantizer.rvq_rest.output_proj.weight']).squeeze(0).T
    q = p0 + pr
    print(f'  RVQ quantized: {q.shape} RMS={q.norm(dim=-1).mean():.2f}')

    # pre_conv + pre-transformer
    x = torch.nn.functional.conv1d(q.T.unsqueeze(0), tw['decoder.pre_conv.conv.weight'], padding=1).squeeze(0).T
    h = x @ tw['decoder.pre_transformer.input_proj.weight'].T

    t3 = time.time()
    for i in range(8):
        p = f'decoder.pre_transformer.layers.{i}'
        hn = rms_norm(h, tw[f'{p}.input_layernorm.weight'])
        q_ = (hn @ tw[f'{p}.self_attn.q_proj.weight'].T).view(seq, 16, 64)
        k_ = (hn @ tw[f'{p}.self_attn.k_proj.weight'].T).view(seq, 16, 64)
        v_ = (hn @ tw[f'{p}.self_attn.v_proj.weight'].T).view(seq, 16, 64)
        for j in range(16): q_[:,j] = rope(q_[:,j], seq); k_[:,j] = rope(k_[:,j], seq)
        sc = torch.einsum('shd,thd->hst', q_, k_) / 8.0
        sc = sc + torch.triu(torch.full((seq, seq), float('-inf')), 1)
        out = torch.einsum('hst,thd->shd', sc.softmax(-1), v_).reshape(seq, -1)
        h = h + out @ tw[f'{p}.self_attn.o_proj.weight'].T
        hn = rms_norm(h, tw[f'{p}.post_attention_layernorm.weight'])
        h = h + (silu(hn @ tw[f'{p}.mlp.gate_proj.weight'].T) * (hn @ tw[f'{p}.mlp.up_proj.weight'].T)) @ tw[f'{p}.mlp.down_proj.weight'].T
    h = rms_norm(h, tw['decoder.pre_transformer.norm.weight'])
    h = h @ tw['decoder.pre_transformer.output_proj.weight'].T
    print(f'  Pre-transformer (8 layers): {h.shape} in {time.time()-t3:.1f}s')

    # Upsample (2 blocks, stride=2)
    x = h.T.unsqueeze(0)
    for b in range(2):
        p = f'decoder.upsample.{b}'
        x = torch.nn.functional.conv_transpose1d(x, tw[f'{p}.0.conv.weight'], stride=2)
        res = x
        x2 = torch.nn.functional.conv1d(x, tw[f'{p}.1.dwconv.conv.weight'], padding=3, groups=1024)
        x2t = x2.transpose(1,2)
        x2t = rms_norm(x2t, tw[f'{p}.1.norm.weight'])
        x2t = gelu(x2t @ tw[f'{p}.1.pwconv1.weight'].T)
        x = res + (x2t @ tw[f'{p}.1.pwconv2.weight'].T).transpose(1,2)
    print(f'  Upsampled: {x.shape}')

    # Main decoder conv stack
    t4 = time.time()
    x = torch.nn.functional.conv1d(x, tw['decoder.decoder.0.conv.weight'], padding=3)
    for bi, (oc, ks, st) in enumerate([(768,16,8),(384,10,5),(192,8,4),(96,6,3)], 1):
        p = f'decoder.decoder.{bi}'
        a = tw.get(f'{p}.block.0.alpha', torch.ones(x.shape[1]))
        x = snake(x, a.view(1,-1,1))
        pad = (ks - st) // 2
        x = torch.nn.functional.conv_transpose1d(x, tw[f'{p}.block.1.conv.weight'], stride=st, padding=pad, output_padding=st-1)
        for r in range(2, 5):
            res = x
            x2 = silu(torch.nn.functional.conv1d(x, tw[f'{p}.block.{r}.conv1.conv.weight'], padding=3))
            x = res + torch.nn.functional.conv1d(x2, tw[f'{p}.block.{r}.conv2.conv.weight'])
        print(f'    Block {bi}: {x.shape}')

    a5 = tw.get('decoder.decoder.5.alpha', torch.ones(96))
    x = snake(x, a5.view(1,-1,1))
    x = torch.tanh(torch.nn.functional.conv1d(x, tw['decoder.decoder.6.conv.weight'], padding=3))
    print(f'  Conv decoder in {time.time()-t4:.1f}s')

    pcm = x.squeeze().detach().numpy()
    duration = len(pcm) / 24000
    out_path = os.path.join(OUT_DIR, 'tts_real_output.wav')
    sf.write(out_path, pcm, 24000)

    print(f'\n=== RESULT ===')
    print(f'  {len(pcm)} samples = {duration:.2f}s at 24kHz')
    print(f'  RMS={abs(pcm).mean():.4f}  peak={abs(pcm).max():.4f}')
    print(f'  WAV: {out_path} ({os.path.getsize(out_path) // 1024} KB)')
    print(f'  Total time: {time.time()-t0:.1f}s')

if __name__ == '__main__':
    main()
