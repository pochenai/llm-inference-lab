from pathlib import Path
import json,os,subprocess,time
while Path('/proc/162376').exists():time.sleep(2)
out=Path('rust/profiling/qwen3_5/cutile-0.5.0-20261109').resolve();commands=json.loads((out/'commands.json').read_text());command=next(c['command'] for c in reversed(commands) if 'triton' in c['command'] and '--result-filename' in c['command']);command=[a for a in command if a!='--disable-cuda-graph']+['--cuda-graph-max-bs','1']
env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_HOME='/usr/local/cuda',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MAX_JOBS='2',TOKENIZERS_PARALLELISM='false');env.pop('LD_LIBRARY_PATH',None);env['PATH']=str(Path(command[0]).parent)+os.pathsep+env['PATH']
for rep in range(3):
 label=f'r{rep}-sglang-graph';run=list(command);run[run.index('--result-filename')+1]=str(out/f'{label}.jsonl');commands.append({'label':label,'command':run});(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');print(label,flush=True)
 with (out/f'{label}.stdout').open('w') as stdout,(out/f'{label}.stderr').open('w') as stderr:subprocess.run(run,env=env,stdout=stdout,stderr=stderr,check=True,timeout=1800)
print('complete',flush=True)
