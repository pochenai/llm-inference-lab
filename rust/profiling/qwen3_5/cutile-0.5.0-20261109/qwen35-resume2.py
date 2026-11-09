import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
while Path('/proc/148802').exists():time.sleep(2)
out=Path('rust/profiling/qwen3_5/cutile-0.5.0-20261109').resolve();out.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_HOME='/usr/local/cuda',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MAX_JOBS='2',TOKENIZERS_PARALLELISM='false');env.pop('LD_LIBRARY_PATH',None)
python=Path('rust/engine/artifacts/sglang-0.5.9-venv/bin/python').absolute();env['PATH']=str(python.parent)+os.pathsep+env['PATH']
commands: list[dict[str,object]]=json.loads((out/'commands.json').read_text())
def run(label: str, command: list[str]) -> None:
 commands.append({'label':label,'command':command});(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');print(label,flush=True)
 with (out/f'{label}.stdout').open('w') as stdout,(out/f'{label}.stderr').open('w') as stderr:subprocess.run(command,env=env,stdout=stdout,stderr=stderr,check=True,timeout=1800)
run('build',['cargo','bench','--manifest-path','rust/Cargo.toml','--offline','--locked','-p','engine','--bench','qwen3_5','--no-run','--message-format=json'])
for line in (out/'build.stdout').read_text().splitlines():
 r=json.loads(line)
 if r.get('reason')=='compiler-artifact' and r['target']['name']=='qwen3_5' and r.get('executable'):shutil.copy2(r['executable'],out/'qwen3_5')
model='/root/autodl-tmp/now/models/Qwen3.5-35B-A3B'
(out/'metadata.json').write_text(json.dumps({'migration_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'model':model,'model_config_sha256':hashlib.sha256((Path(model)/'config.json').read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256((out/'qwen3_5').read_bytes()).hexdigest(),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],text=True),'boundary':'Rust includes host token and complete-logit validation; SGLang uses GPU sampling; both eager, fixed output count, fresh state'},indent=2)+'\n')
shutil.copy2('rust/engine/benches/qwen3_5.rs',out/'qwen3_5.rs');shutil.copy2('rust/engine/benches/sglang_qwen2.py',out/'sglang_adapter.py')
for rep in range(3):
 for engine in (['rust','sglang'] if rep%2==0 else ['sglang','rust']):
  if rep==0 and engine=='rust':continue
  if engine=='rust':command=[str(out/'qwen3_5'),'--model-dir',model,'--tokens','32','--output-tokens','8','--warmup','2','--iterations','3']
  else:command=[str(python),str(out/'sglang_adapter.py'),'--model-path',model,'--attention-backend','triton','--dtype','bfloat16','--mem-fraction-static','0.9','--max-total-tokens','128','--max-running-requests','1','--max-mamba-cache-size','4','--mamba-ssm-dtype','float32','--context-length','512','--batch-size','1','--input-len','32','--output-len','8','--comparison-warmup','2','--comparison-iterations','3','--disable-cuda-graph','--disable-radix-cache','--result-filename',str(out/f'r{rep}-sglang.jsonl')]
  run(f'r{rep}-{engine}',command)
print('complete',flush=True)
