from pathlib import Path
import hashlib,json,os,shutil,subprocess,time
while Path('/proc/163781').exists():time.sleep(2)
out=Path('rust/profiling/qwen3_5/cutile-0.5.0-20261109').resolve();env=dict(os.environ,CUDA_HOME='/usr/local/cuda');env.pop('LD_LIBRARY_PATH',None)
with (out/'profile-build.jsonl').open('w') as stdout,(out/'profile-build.stderr').open('w') as stderr:subprocess.run(['cargo','bench','--manifest-path','rust/Cargo.toml','--offline','--locked','-p','engine','--bench','qwen3_5','--no-run','--message-format=json'],env=env,stdout=stdout,stderr=stderr,check=True)
for line in (out/'profile-build.jsonl').read_text().splitlines():
 r=json.loads(line)
 if r.get('reason')=='compiler-artifact' and r['target']['name']=='qwen3_5' and r.get('executable'):shutil.copy2(r['executable'],out/'qwen3_5-profile')
shutil.copy2('rust/engine/benches/qwen3_5.rs',out/'qwen3_5-profile.rs')
for flags,expected in [(['--tokens','0'],'tokens and iterations must be positive'),(['--output-tokens','1'],'tokens and iterations must be positive'),(['--iterations','0'],'tokens and iterations must be positive'),(['--warmup','18446744073709551615','--iterations','1'],'sample count overflow'),(['--tokens','18446744073709551615'],'token capacity overflow')]:
 result=subprocess.run([str(out/'qwen3_5-profile'),'--model-dir','/unused-invalid-model',*flags],env=env,capture_output=True,text=True);assert result.returncode!=0 and expected in result.stderr
(out/'invalid-args.json').write_text(json.dumps({'passed':5})+'\n')
command=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true',f'--output={out/"rust-profile"}',str(out/'qwen3_5-profile'),'--model-dir','/root/autodl-tmp/now/models/Qwen3.5-35B-A3B','--tokens','32','--output-tokens','8','--warmup','2','--iterations','1','--profile'];(out/'profile-command.json').write_text(json.dumps(command,indent=2)+'\n');print('profile',flush=True)
with (out/'rust-profile.stdout').open('w') as stdout,(out/'rust-profile.stderr').open('w') as stderr:subprocess.run(command,env=env,stdout=stdout,stderr=stderr,check=True,timeout=1200)
with (out/'rust-profile-stats.csv').open('w') as stdout:subprocess.run(['nsys','stats','--report','cuda_api_sum,cuda_gpu_kern_sum','--format','csv',str(out/'rust-profile.nsys-rep')],env=env,stdout=stdout,check=True)
print('complete',flush=True)
