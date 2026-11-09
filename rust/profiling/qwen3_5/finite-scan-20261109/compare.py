"""Sequential, rotated native BF16 graph comparison; retain every raw sample."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import statistics
import subprocess

out = Path(__file__).absolute().parent
baseline = Path(os.environ['QWEN35_BASELINE']).absolute()
candidate = Path(os.environ['QWEN35_CANDIDATE']).absolute()
env = dict(os.environ, CUDA_HOME='/usr/local/cuda')
env.pop('LD_LIBRARY_PATH', None)
for label, source in [('baseline', baseline), ('candidate', candidate)]:
    target = out / label
    if source != target:
        shutil.copy2(source, target)
metadata = {'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), 'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'], text=True), 'hashes': {label: hashlib.sha256((out/label).read_bytes()).hexdigest() for label in ['baseline', 'candidate']}, 'commands': []}
metadata['environment'] = {name: env.get(name) for name in ['CUDA_HOME', 'CUDA_ASYNC_HOST_SYNC', 'CUDA_ASYNC_SPIN_BUDGET_US']}
(out/'patch.diff').write_bytes(subprocess.check_output(['git', 'diff', '--', 'rust/engine/src']))
all_results = []
for repetition in range(3):
    for label in (['baseline', 'candidate'] if repetition % 2 == 0 else ['candidate', 'baseline']):
        name = f'r{repetition}-{label}'
        command = ['taskset', '-c', '24-31', str(out/label), '--model-dir', '/root/autodl-tmp/now/models/Qwen3.5-35B-A3B', '--tokens', '32', '--output-tokens', '8', '--warmup', '2', '--iterations', '3', '--cuda-graph']
        metadata['commands'].append(command)
        (out/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        print(name, flush=True)
        with (out/(name+'.jsonl')).open('w') as stdout, (out/(name+'.stderr')).open('w') as stderr:
            subprocess.run(command, env=env, stdout=stdout, stderr=stderr, check=True, timeout=1200)
        records = [json.loads(line) for line in (out/(name+'.jsonl')).read_text().splitlines() if line.startswith('{')]
        samples = [r for r in records if r.get('event') == 'sample' and not r['warmup']]
        assert len(samples) == 3
        result = {'variant': label, 'repetition': repetition, 'prefill_ms': statistics.median(r['prefill_seconds']*1000 for r in samples), 'decode_ms': statistics.median(statistics.mean(r['decode_step_seconds'])*1000 for r in samples), 'total_ms': statistics.median((r['prefill_seconds']+sum(r['decode_step_seconds']))*1000 for r in samples), 'generated_ids': samples[0]['generated_ids']}
        all_results.append(result)
        print(result, flush=True)
        summary = {'processes': all_results, 'median': {v: {metric: statistics.median(r[metric] for r in all_results if r['variant'] == v) for metric in ['prefill_ms', 'decode_ms', 'total_ms']} for v in sorted({r['variant'] for r in all_results})}}
        (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print('complete', flush=True)
