"""Native paired control; profiler and module hooks are disabled."""
from pathlib import Path
import json
import os
import subprocess

out = Path('rust/profiling/qwen3_5/sglang-gap-20261108').resolve()
python = Path('rust/engine/artifacts/sglang-0.5.9-venv/bin/python').absolute()
env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_HOME='/usr/local/cuda', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MAX_JOBS='2', TOKENIZERS_PARALLELISM='false')
env.pop('LD_LIBRARY_PATH', None)
env['PATH'] = str(python.parent) + os.pathsep + env['PATH']
commands: list[dict[str, object]] = []
for repetition in range(3):
    for variant in (['default', 'serial'] if repetition % 2 == 0 else ['serial', 'default']):
        script = str(out/'sglang_serial_moe.py') if variant == 'serial' else 'rust/engine/benches/sglang_qwen2.py'
        label = f'overlap-r{repetition}-{variant}'
        command = [str(python), script, '--model-path', '/root/autodl-tmp/now/models/Qwen3.5-35B-A3B', '--attention-backend', 'triton', '--dtype', 'bfloat16', '--mem-fraction-static', '0.9', '--max-total-tokens', '128', '--max-running-requests', '1', '--max-mamba-cache-size', '4', '--mamba-ssm-dtype', 'float32', '--context-length', '512', '--batch-size', '1', '--input-len', '32', '--output-len', '8', '--comparison-warmup', '2', '--comparison-iterations', '3', '--disable-radix-cache', '--cuda-graph-max-bs', '1', '--result-filename', str(out/(label+'.jsonl'))]
        commands.append({'label': label, 'command': command})
        (out/'overlap-commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        print(label, flush=True)
        with (out/(label+'.stdout')).open('w') as stdout, (out/(label+'.stderr')).open('w') as stderr:
            subprocess.run(command, env=env, stdout=stdout, stderr=stderr, check=True, timeout=1800)
print('complete', flush=True)
