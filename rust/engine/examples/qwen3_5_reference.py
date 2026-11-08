"""Independent eager PyTorch text reference; no Transformers or custom GPU kernels.

Equations follow Hugging Face Qwen3.5 MoE and NVIDIA TileGym's GDN reference.
Load one decoder layer at a time so BF16 reference and Rust can run separately.
"""
from __future__ import annotations

import argparse
import json
import math
import mmap
import shutil
import struct
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


class Weights:
    def __init__(self, directory: Path, device: str) -> None:
        self.directory = directory
        self.device = device
        index = directory / "model.safetensors.index.json"
        self.index: dict[str, str] = json.loads(index.read_text())["weight_map"] if index.exists() else {}

    def get(self, name: str) -> torch.Tensor:
        path = self.directory / self.index.get(name, "model.safetensors")
        with path.open("rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_COPY) as mapped:
            length = struct.unpack("<Q", mapped[:8])[0]
            header = json.loads(mapped[8:8 + length])
            info = header[name]
            dtype = {"BF16": torch.bfloat16, "F32": torch.float32}[info["dtype"]]
            start, end = info["data_offsets"]
            data = torch.frombuffer(mapped, dtype=dtype, offset=8 + length + start, count=(end-start)//dtype.itemsize).reshape(info["shape"]).clone()
        return data.to(self.device)


def save_weights(path: Path, weights: dict[str, torch.Tensor]) -> None:
    header: dict[str, Any] = {}
    chunks: list[bytes] = []
    offset = 0
    for name, tensor in weights.items():
        chunk = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        header[name] = {"dtype": "BF16" if tensor.dtype == torch.bfloat16 else "F32", "shape": list(tensor.shape), "data_offsets": [offset, offset+len(chunk)]}
        chunks.append(chunk)
        offset += len(chunk)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    with path.open("wb") as file:
        file.write(struct.pack("<Q", len(encoded)))
        file.write(encoded)
        for chunk in chunks:
            file.write(chunk)


def tiny(directory: Path, tokenizer_source: Path) -> None:
    artifacts = Path(__file__).resolve().parent.parent / "artifacts" / "qwen3_5"
    if not directory.resolve().is_relative_to(artifacts):
        raise ValueError("Tiny fixtures must be written under rust/engine/artifacts/qwen3_5")
    directory.mkdir(parents=True, exist_ok=True)
    config = json.loads((tokenizer_source / "config.json").read_text())
    c = config["text_config"]
    c.update(hidden_size=128, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
             linear_num_key_heads=1, linear_num_value_heads=2, num_experts=4, num_experts_per_tok=2,
             moe_intermediate_size=128, shared_expert_intermediate_size=256,
             layer_types=["linear_attention"]*3+["full_attention"])
    (directory/"config.json").write_text(json.dumps(config))
    shutil.copyfile(tokenizer_source/"tokenizer.json", directory/"tokenizer.json")
    torch.manual_seed(20261108)
    weights: dict[str, torch.Tensor] = {}

    def add(name: str, shape: tuple[int, ...], scale: float = 0.02, dtype: torch.dtype = torch.bfloat16) -> None:
        weights[name] = (torch.randn(shape)*scale).to(dtype)

    add("model.language_model.embed_tokens.weight", (c["vocab_size"], 128))
    add("lm_head.weight", (c["vocab_size"], 128))
    add("model.language_model.norm.weight", (128,))
    for layer, kind in enumerate(c["layer_types"]):
        p = f"model.language_model.layers.{layer}"
        add(f"{p}.input_layernorm.weight", (128,))
        add(f"{p}.post_attention_layernorm.weight", (128,))
        if kind == "linear_attention":
            for name, shape in {"in_proj_qkv.weight": (512,128), "in_proj_z.weight": (256,128),
                                "in_proj_a.weight": (2,128), "in_proj_b.weight": (2,128),
                                "out_proj.weight": (128,256), "conv1d.weight": (512,1,4), "dt_bias": (2,)}.items():
                add(f"{p}.linear_attn.{name}", shape)
            add(f"{p}.linear_attn.A_log", (2,), dtype=torch.float32)
            weights[f"{p}.linear_attn.norm.weight"] = torch.ones(128)
        else:
            for name, shape in {"q_proj.weight": (1024,128), "k_proj.weight": (256,128), "v_proj.weight": (256,128),
                                "o_proj.weight": (128,512), "q_norm.weight": (256,), "k_norm.weight": (256,)}.items():
                add(f"{p}.self_attn.{name}", shape)
        for name, shape in {"gate.weight": (4,128), "experts.gate_up_proj": (4,256,128), "experts.down_proj": (4,128,128),
                            "shared_expert.gate_proj.weight": (256,128), "shared_expert.up_proj.weight": (256,128),
                            "shared_expert.down_proj.weight": (128,256), "shared_expert_gate.weight": (1,128)}.items():
            add(f"{p}.mlp.{name}", shape)
    save_weights(directory/"model.safetensors", weights)


@torch.inference_mode()
def reference(directory: Path, ids: list[int], device: str, trace_dir: Path | None = None) -> torch.Tensor:
    c = json.loads((directory/"config.json").read_text())["text_config"]
    weights = Weights(directory, device)
    x = F.embedding(torch.tensor(ids, device=device), weights.get("model.language_model.embed_tokens.weight"))
    eps = c["rms_norm_eps"]
    def capture(tensor: torch.Tensor, index: int, tag: str) -> None:
        if trace_dir is not None:
            trace_dir.mkdir(parents=True,exist_ok=True)
            (trace_dir/f"{index:02}-{tag}.raw").write_bytes(tensor.contiguous().cpu().view(torch.uint8).numpy().tobytes())

    def norm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+eps)*(1+w.float())).to(x.dtype)

    for layer, kind in enumerate(c["layer_types"]):
        p = f"model.language_model.layers.{layer}"
        n = norm(x, weights.get(f"{p}.input_layernorm.weight"))
        capture(n,layer,"norm")
        if kind == "linear_attention":
            qkv = F.linear(n, weights.get(f"{p}.linear_attn.in_proj_qkv.weight"))
            channels = qkv.shape[-1]
            # Accumulate convolution and SiLU before the BF16 store, like causal_conv1d.
            convolved = F.silu(F.conv1d(qkv.float().T.unsqueeze(0), weights.get(f"{p}.linear_attn.conv1d.weight").float(), padding=3, groups=channels)[0,:,:len(ids)].T).to(x.dtype)
            kh, vh = c["linear_num_key_heads"], c["linear_num_value_heads"]
            q,k,v = convolved.split([kh*128,kh*128,vh*128],dim=-1)
            q,k = [t.reshape(-1,kh,128).float().repeat_interleave(vh//kh,dim=1) for t in (q,k)]
            q = q*torch.rsqrt(q.square().sum(-1,keepdim=True)+1e-6)/math.sqrt(128)
            k = k*torch.rsqrt(k.square().sum(-1,keepdim=True)+1e-6)
            v = v.reshape(-1,vh,128).float()
            a = F.linear(n,weights.get(f"{p}.linear_attn.in_proj_a.weight")).float()
            b = F.linear(n,weights.get(f"{p}.linear_attn.in_proj_b.weight")).sigmoid().float()
            g = -weights.get(f"{p}.linear_attn.A_log").float().exp()*F.softplus(a+weights.get(f"{p}.linear_attn.dt_bias").float())
            state = torch.zeros(vh,128,128,device=device)
            outputs: list[torch.Tensor] = []
            for t in range(len(ids)):
                state *= g[t].exp()[:,None,None]
                delta = (v[t]-(state*k[t,:, :,None]).sum(-2))*b[t,:,None]
                state += k[t,:,:,None]*delta[:,None,:]
                outputs.append((state*q[t,:,:,None]).sum(-2))
            core = torch.stack(outputs).to(x.dtype).float()
            core = (core*torch.rsqrt(core.square().mean(-1,keepdim=True)+eps)).to(x.dtype)
            z = F.linear(n,weights.get(f"{p}.linear_attn.in_proj_z.weight")).reshape(-1,vh,128)
            core = (core*weights.get(f"{p}.linear_attn.norm.weight")*F.silu(z.float())).to(x.dtype)
            mixed = F.linear(core.flatten(1),weights.get(f"{p}.linear_attn.out_proj.weight"))
        else:
            qh, kvh, d = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
            q, gate = F.linear(n,weights.get(f"{p}.self_attn.q_proj.weight")).reshape(-1,qh,2*d).chunk(2,-1)
            q = norm(q,weights.get(f"{p}.self_attn.q_norm.weight"))
            k = norm(F.linear(n,weights.get(f"{p}.self_attn.k_proj.weight")).reshape(-1,kvh,d),weights.get(f"{p}.self_attn.k_norm.weight"))
            v = F.linear(n,weights.get(f"{p}.self_attn.v_proj.weight")).reshape(-1,kvh,d)
            freq = c["rope_parameters"]["rope_theta"]**(-torch.arange(0,64,2,device=device).float()/64)
            angle = torch.arange(len(ids),device=device).float()[:,None]*freq
            co = torch.cat([angle,angle],-1).cos().to(x.dtype)[:,None,:]
            si = torch.cat([angle,angle],-1).sin().to(x.dtype)[:,None,:]
            def rope(t: torch.Tensor) -> torch.Tensor:
                first = t[...,:64]
                rotated = torch.cat([-first[...,32:],first[...,:32]],-1)
                return torch.cat([first*co+rotated*si,t[...,64:]],-1)
            q,k = rope(q),rope(k)
            k,v = [t.repeat_interleave(qh//kvh,dim=1) for t in (k,v)]
            scores = torch.einsum("thd,shd->hts",q.float(),k.float())/math.sqrt(d)
            mask = torch.ones(len(ids),len(ids),device=device,dtype=torch.bool).triu(1)
            scores.masked_fill_(mask, -torch.inf)
            attended = torch.einsum("hts,shd->thd",scores.softmax(-1),v.float()).to(x.dtype)
            mixed = F.linear((attended*gate.sigmoid()).flatten(1),weights.get(f"{p}.self_attn.o_proj.weight"))
        capture(mixed,layer,"mixed")
        x = x+mixed
        n = norm(x, weights.get(f"{p}.post_attention_layernorm.weight"))
        router = F.linear(n,weights.get(f"{p}.mlp.gate.weight"))
        capture(router,layer,"router")
        # torch.topk does not promise stable tied indices; match the engine's
        # explicit lowest-index tie policy with an independent stable sort.
        selected = router.float().argsort(dim=-1,descending=True,stable=True)[:,:c["num_experts_per_tok"]]
        scores = router.float().softmax(-1).gather(-1,selected)
        scores = (scores/scores.sum(-1,keepdim=True)).to(x.dtype)
        gu = weights.get(f"{p}.mlp.experts.gate_up_proj")
        down = weights.get(f"{p}.mlp.experts.down_proj")
        routed = torch.zeros_like(n,dtype=torch.float32)
        for token in range(len(ids)):
            for rank in range(c["num_experts_per_tok"]):
                expert = selected[token,rank].item()
                g,u = F.linear(n[token],gu[expert]).chunk(2,-1)
                out = F.linear(F.silu(g)*u,down[expert])
                routed[token] += (out*scores[token,rank]).float()
        g = F.linear(n,weights.get(f"{p}.mlp.shared_expert.gate_proj.weight"))
        u = F.linear(n,weights.get(f"{p}.mlp.shared_expert.up_proj.weight"))
        shared = F.linear(F.silu(g)*u,weights.get(f"{p}.mlp.shared_expert.down_proj.weight"))
        shared *= F.linear(n,weights.get(f"{p}.mlp.shared_expert_gate.weight")).sigmoid()
        x = x + (routed.to(x.dtype)+shared)
        capture(x,layer,"hidden")
        print(f"reference layer {layer+1}/{len(c['layer_types'])}",flush=True)
    return F.linear(norm(x[-1:],weights.get("model.language_model.norm.weight")).float(),weights.get("lm_head.weight").float())[0].cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir",type=Path,required=True)
    parser.add_argument("--tiny-from",type=Path)
    parser.add_argument("--ids",default="1,7,9,3,12")
    parser.add_argument("--device",default="cpu",choices=["cpu","cuda"])
    parser.add_argument("--trace-dir",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    # The Rust projections accumulate BF16 products in FP32 without truncated
    # partial reductions. Keep the independent reference on the same contract.
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.tiny_from:
        tiny(args.model_dir,args.tiny_from)
    ids = [int(s) for s in args.ids.split(",")]
    logits = reference(args.model_dir,ids,args.device,args.trace_dir)
    args.output.write_bytes(logits.numpy().tobytes())
    print(json.dumps({"ids":ids,"argmax":logits.argmax().item(),"max_logit":logits.max().item()}))


if __name__ == "__main__":
    main()
