"""Collect sequential graph traces and a separate eager attribution trace."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess

out = Path('rust/profiling/qwen3_5/sglang-gap-20261108').resolve()
model = '/root/autodl-tmp/now/models/Qwen3.5-35B-A3B'
python = Path('rust/engine/artifacts/sglang-0.5.9-venv/bin/python').absolute()
env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_HOME='/usr/local/cuda', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MAX_JOBS='2', TOKENIZERS_PARALLELISM='false', QWEN35_PROFILE_DIR=str(out))
env.pop('LD_LIBRARY_PATH', None)
env['PATH'] = str(python.parent) + os.pathsep + env['PATH']
for line in Path('/tmp/qwen35-gap-build.jsonl').read_text().splitlines():
    record = json.loads(line)
    if record.get('reason') == 'compiler-artifact' and record['target']['name'] == 'qwen3_5' and record.get('executable'):
        shutil.copy2(record['executable'], out / 'rust-graph')
commands: list[dict[str, object]] = []
(out / 'metadata.json').write_text(json.dumps({'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), 'rust_binary_sha256': hashlib.sha256((out / 'rust-graph').read_bytes()).hexdigest(), 'model': model, 'cpu_affinity': sorted(os.sched_getaffinity(0)), 'native_reference': '../cuda-graph-20261108/summary.json', 'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'], text=True)}, indent=2)+'\n')
for variant in ['sglang-graph', 'rust-graph', 'sglang-eager']:
    if variant == 'rust-graph':
        executable = [str(out/'rust-graph'), '--model-dir', model, '--tokens', '32', '--output-tokens', '8', '--warmup', '2', '--iterations', '1', '--profile', '--cuda-graph']
    else:
        executable = [str(python), str(out/'sglang_profile.py'), '--model-path', model, '--attention-backend', 'triton', '--dtype', 'bfloat16', '--mem-fraction-static', '0.9', '--max-total-tokens', '128', '--max-running-requests', '1', '--max-mamba-cache-size', '4', '--mamba-ssm-dtype', 'float32', '--context-length', '512', '--batch-size', '1', '--input-len', '32', '--output-len', '8', '--comparison-warmup', '2', '--comparison-iterations', '1', '--disable-radix-cache', '--result-filename', str(out/(variant+'.jsonl'))]
        executable += ['--disable-cuda-graph'] if variant.endswith('eager') else ['--cuda-graph-max-bs', '1']
    command = ['nsys', 'profile', '--trace=cuda,nvtx', '--cuda-graph-trace=node', '--sample=none', '--cpuctxsw=none', '--capture-range=cudaProfilerApi', '--capture-range-end=stop', '--force-overwrite=true', '--output='+str(out/variant), *executable]
    commands.append({'label': variant, 'command': command})
    (out/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
    print(variant, flush=True)
    with (out/(variant+'.stdout')).open('w') as stdout, (out/(variant+'.stderr')).open('w') as stderr:
        subprocess.run(command, env=env, stdout=stdout, stderr=stderr, check=True, timeout=1800)
    with (out/(variant+'-stats.csv')).open('w') as stdout:
        subprocess.run(['nsys', 'stats', '--report', 'cuda_api_sum,cuda_gpu_kern_sum,nvtx_sum', '--format', 'csv', str(out/(variant+'.nsys-rep'))], env=env, stdout=stdout, check=True)
print('complete', flush=True)
