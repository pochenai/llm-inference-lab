from pathlib import Path
import hashlib,json,os,shutil,subprocess,time
while Path('/proc/175980').exists():time.sleep(2)
assert 'test result: ok.' in Path('/tmp/qwen35-workspace-bf16-final.log').read_text()
out=Path('rust/profiling/qwen3_5/workspace-reuse-20261109').resolve();out.mkdir(parents=True,exist_ok=False)
base=Path('rust/profiling/qwen3_5/cutile-0.5.0-20261109')
shutil.copy2(base/'qwen3_5-profile',out/'baseline')
for line in Path('/tmp/qwen35-workspace-build.jsonl').read_text().splitlines():
 r=json.loads(line)
 if r.get('reason')=='compiler-artifact' and r['target']['name']=='qwen3_5' and r.get('executable'):shutil.copy2(r['executable'],out/'candidate')
for name in ['engine.rs','workspace.rs','state.rs','mod.rs']:shutil.copy2(Path('rust/engine/src/runner/model/qwen3_5')/name,out/name)
shutil.copy2('rust/engine/benches/qwen3_5.rs',out/'qwen3_5.rs')
shutil.copy2(base/'sglang_adapter.py',out/'sglang_adapter.py')
env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_HOME='/usr/local/cuda',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MAX_JOBS='2',TOKENIZERS_PARALLELISM='false');env.pop('LD_LIBRARY_PATH',None)
python=Path('rust/engine/artifacts/sglang-0.5.9-venv/bin/python').absolute();env['PATH']=str(python.parent)+os.pathsep+env['PATH']
model='/root/autodl-tmp/now/models/Qwen3.5-35B-A3B';commands: list[dict[str,object]]=[]
def run(label: str, command: list[str]) -> None:
 commands.append({'label':label,'command':command});(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');print(label,flush=True)
 with (out/f'{label}.stdout').open('w') as stdout,(out/f'{label}.stderr').open('w') as stderr:subprocess.run(command,env=env,stdout=stdout,stderr=stderr,check=True,timeout=1800)
(out/'metadata.json').write_text(json.dumps({'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'model':model,'dtype':'bf16','ssm_state_dtype':'float32','binaries':{v:hashlib.sha256((out/v).read_bytes()).hexdigest() for v in ['baseline','candidate']},'model_config_sha256':hashlib.sha256((Path(model)/'config.json').read_bytes()).hexdigest(),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],text=True)},indent=2)+'\n')
engines=['baseline','candidate','sglang-graph']
for rep in range(3):
 for engine in engines[rep:]+engines[:rep]:
  if engine!='sglang-graph':command=[str(out/engine),'--model-dir',model,'--tokens','32','--output-tokens','8','--warmup','2','--iterations','3']
  else:command=[str(python),str(out/'sglang_adapter.py'),'--model-path',model,'--attention-backend','triton','--dtype','bfloat16','--mem-fraction-static','0.9','--max-total-tokens','128','--max-running-requests','1','--max-mamba-cache-size','4','--mamba-ssm-dtype','float32','--context-length','512','--batch-size','1','--input-len','32','--output-len','8','--comparison-warmup','2','--comparison-iterations','3','--cuda-graph-max-bs','1','--disable-radix-cache','--result-filename',str(out/f'r{rep}-{engine}.jsonl')]
  run(f'r{rep}-{engine}',command)
print('native complete',flush=True)
for variant in ['baseline','candidate']:
 label=f'profile-{variant}'
 run(label,['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true',f'--output={out/label}',str(out/variant),'--model-dir',model,'--tokens','32','--output-tokens','8','--warmup','2','--iterations','1','--profile'])
 with (out/f'{label}-stats.csv').open('w') as stdout:subprocess.run(['nsys','stats','--report','cuda_api_sum,cuda_gpu_kern_sum,nvtx_sum','--format','csv',str(out/f'{label}.nsys-rep')],env=env,stdout=stdout,check=True)
print('complete',flush=True)
