"""Create an offline CPU/FP32 reference for the cuTile Qwen2 example."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--prompt", default="Say hello.")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": args.prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    history = list(ids)
    generated: list[int] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for step in range(args.max_new_tokens):
            logits = model(torch.tensor([history]), use_cache=False).logits[0, -1].float()
            if step == 0:
                (args.output_dir / "logits.f32").write_bytes(logits.numpy().tobytes())
            token = int(logits.argmax().item())
            generated.append(token)
            history.append(token)
            if token in [151643, 151645]:
                break
    result: dict[str, object] = {
        "model_dir": str(args.model_dir.resolve()), "dtype": "float32", "prompt": prompt,
        "input_ids": ids, "generated_ids": generated,
        "text": tokenizer.decode(generated, skip_special_tokens=True),
    }
    (args.output_dir / "reference.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
