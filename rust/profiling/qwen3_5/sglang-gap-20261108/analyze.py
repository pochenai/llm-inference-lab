"""Attribute paired graph traces without conflating overlapping kernel sums with wall time."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from statistics import mean, median
from typing import Any

OUT = Path('rust/profiling/qwen3_5/sglang-gap-20261108')


@dataclass
class Kernel:
    start: int
    end: int
    name: str
    stream: int
    correlation: int
    grid: tuple[int, int, int]
    block: tuple[int, int, int]


def read_kernels(connection: sqlite3.Connection) -> list[Kernel]:
    rows = connection.execute('SELECT k.start,k.end,s.value,k.streamId,k.correlationId,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start').fetchall()
    return [Kernel(a,b,n,st,c,(gx,gy,gz),(bx,by,bz)) for a,b,n,st,c,gx,gy,gz,bx,by,bz in rows]


def ranges(connection: sqlite3.Connection, prefix: str) -> list[tuple[str, int, int, int]]:
    rows = connection.execute('SELECT COALESCE(n.text,s.value),n.start,n.end,n.globalTid,n.endGlobalTid FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId').fetchall()
    selected = [(n,a,b,t) for n,a,b,t,et in rows if n and n.startswith(prefix)]
    assert all(b is not None and b >= a and (et is None or et == t) for n,a,b,t,et in rows if n and n.startswith(prefix))
    return sorted(selected, key=lambda r: r[1])


def union_ns(kernels: list[Kernel]) -> int:
    merged = 0
    lo, hi = kernels[0].start, kernels[0].end
    for kernel in kernels[1:]:
        if kernel.start > hi:
            merged += hi-lo
            lo,hi = kernel.start,kernel.end
        else:
            hi = max(hi,kernel.end)
    return merged + hi-lo


def family(name: str) -> str:
    if name == 'linear_entry' or any(word in name for word in ('gemvx::', 'cublasLt::splitKreduce', 'cutlass::Kernel')):
        return 'dense_projections'
    if name in ('expert_entry', 'fused_moe_kernel'):
        return 'routed_expert_gemm'
    if name in ('gqa_graph_entry', '_fwd_grouped_kernel_stage1', '_fwd_kernel_stage2'):
        return 'gqa_attention'
    return 'other'


def summarize(steps: list[list[Kernel]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    times: Counter[str] = Counter()
    families: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    per_step: list[dict[str, float | int]] = []
    for kernels in steps:
        total = sum(k.end-k.start for k in kernels)
        busy = union_ns(kernels)
        span = max(k.end for k in kernels)-kernels[0].start
        per_step.append({'kernels':len(kernels),'kernel_sum_ms':total/1e6,'kernel_union_ms':busy/1e6,'overlap_ms':(total-busy)/1e6,'kernel_span_ms':span/1e6,'span_non_kernel_ms':(span-busy)/1e6})
        for k in kernels:
            counts[k.name] += 1
            times[k.name] += k.end-k.start
            families[family(k.name)] += k.end-k.start
            family_counts[family(k.name)] += 1
    return {'per_step':per_step,'mean':{key:mean(s[key] for s in per_step) for key in per_step[0]},'families':{key:{'ms_per_step':value/len(steps)/1e6,'kernels_per_step':family_counts[key]/len(steps)} for key,value in families.items()},'kernels':[{'name':name,'count_per_step':counts[name]/len(steps),'ms_per_step':value/len(steps)/1e6} for name,value in times.most_common()]}


results: dict[str, Any] = {}
for variant in ['rust-graph', 'sglang-graph']:
    connection = sqlite3.connect(OUT/(variant+'.sqlite'))
    kernels = read_kernels(connection)
    labels = ranges(connection, 'qwen3_5.' if variant.startswith('rust') else 'sglang.')
    if variant == 'rust-graph':
        groups: list[list[Kernel]] = []
        current: list[Kernel] = []
        for kernel in kernels:
            current.append(kernel)
            if kernel.name == 'argmax_final_entry':
                groups.append(current)
                current=[]
        assert len(groups)==8 and not current
        steps=groups[1:]
        api=connection.execute('SELECT a.start,a.end,s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME a JOIN StringIds s ON s.id=a.nameId ORDER BY a.start').fetchall()
        replay=[]
        for name,start,end,tid in labels:
            if name!='qwen3_5.decode.graph_replay':
                continue
            uploads=[(a,b) for a,b,n in api if start<=a<end and n.startswith('cuMemcpyHtoDAsync')]
            launches=[(a,b) for a,b,n in api if start<=a<end and n=='cuGraphLaunch']
            assert len(uploads)==2 and len(launches)==1
            replay.append({'metadata_preparation_ms':(uploads[0][0]-start)/1e6,'post_upload_launch_preparation_ms':(launches[0][0]-uploads[-1][1])/1e6,'driver_graph_launch_ms':(launches[0][1]-launches[0][0])/1e6,'replay_submit_range_ms':(end-start)/1e6})
        results['rust_replay_cpu']=replay
        results['rust_validation_ms']=[(b-a)/1e6 for n,a,b,t in labels if n=='qwen3_5.host_validation'][1:]
        validation_ranges=[(a,b) for n,a,b,t in labels if n=='qwen3_5.host_validation']
        results['rust_post_readback_tail_ms']=[]
        for group,(validation_start,validation_end) in zip(groups[1:],validation_ranges[1:]):
            ends=connection.execute('SELECT end FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind=2 AND start>=? AND end<=?',(group[-1].end,validation_start)).fetchall()
            assert len(ends)==2
            results['rust_post_readback_tail_ms'].append((validation_start-max(row[0] for row in ends))/1e6)

        copies=connection.execute('SELECT copyKind,COUNT(*),SUM(bytes),SUM(end-start)/1e6 FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind').fetchall()
        results['rust_copies_cohort']=[{'kind':kind,'count':count,'bytes':size,'gpu_ms':duration} for kind,count,size,duration in copies]
    else:
        starts=[a for n,a,b,t in labels if n=='sglang.decode']
        assert len(starts)==7 and sum(n=='sglang.prefill' for n,*_ in labels)==1
        steps=[[k for k in kernels if start<=k.start<(starts[index+1] if index<6 else 10**30)] for index,start in enumerate(starts)]
        assert all(len(step)==1100 for step in steps)
    results[variant]=summarize(steps)
    connection.close()

# Eager hooks assign mathematical roles; their timings are diagnostic, not graph latency.
c=sqlite3.connect(OUT/'sglang-eager.sqlite')
labels=ranges(c,'sglang.')
start=min(a for n,a,b,t in labels if n=='sglang.decode')
modules=[(n,a,b,t) for n,a,b,t in labels if n.startswith('sglang.module.')]
api={corr:(a,t) for a,t,corr in c.execute('SELECT start,globalTid,correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME')}
roles: dict[str, dict[str, Any]]={}
for k in read_kernels(c):
    if k.start<start or family(k.name)!='dense_projections':
        continue
    if k.correlation not in api:
        continue
    a,t=api[k.correlation]
    matches=[(n,b-a0) for n,a0,b,tid in modules if tid==t and a0<=a<b]
    label=min(matches,key=lambda x:x[1])[0] if matches else 'unhooked'
    if '.layers.' in label:
        label=label.split('.layers.',1)[1].split('.',1)[1]
    item=roles.setdefault(label,{'kernels':0,'gpu_ns':0,'kernel_names':set()})
    item['kernels']+=1;item['gpu_ns']+=k.end-k.start;item['kernel_names'].add(k.name)
results['sglang_eager_dense_roles']={name:{'kernels_per_step':r['kernels']/7,'ms_per_step':r['gpu_ns']/7/1e6,'kernel_names':sorted(r['kernel_names'])} for name,r in roles.items()}
c.close()
# Map dense graph kernel roles from the eager trace by identical symbol/geometry and
# within-key occurrence order. Main-stream projections preserve order; alternate
# stream router projections have distinct keys. Compare timings from graph only.
def key(kernel: Kernel) -> tuple[str, tuple[int,int,int], tuple[int,int,int]]:
    return kernel.name, kernel.grid, kernel.block


def module_role(label: str) -> str:
    if '.layers.' in label:
        return label.split('.layers.',1)[1].split('.',1)[1]
    return 'lm_head' if label=='unhooked' else label


c=sqlite3.connect(OUT/'sglang-eager.sqlite')
rr=ranges(c,'sglang.')
starts=[a for n,a,b,t in rr if n=='sglang.decode']
mods=[r for r in rr if r[0].startswith('sglang.module.')]
api={corr:(a,t) for a,t,corr in c.execute('SELECT start,globalTid,correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME')}
template: dict[tuple[str,tuple[int,int,int],tuple[int,int,int]],list[str]]=defaultdict(list)
for k in read_kernels(c):
    if not starts[3]<=k.start<starts[4] or family(k.name)!='dense_projections':continue
    a,t=api[k.correlation]
    matches=[(n,b-a0) for n,a0,b,tid in mods if tid==t and a0<=a<b]
    label=min(matches,key=lambda x:x[1])[0] if matches else 'unhooked'
    template[key(k)].append(module_role(label))
c.close()
c=sqlite3.connect(OUT/'sglang-graph.sqlite');rr=ranges(c,'sglang.');starts=[a for n,a,b,t in rr if n=='sglang.decode'];kernels=read_kernels(c)
role_times: Counter[str]=Counter();role_counts: Counter[str]=Counter()
for index,start in enumerate(starts):
    selected=[k for k in kernels if start<=k.start<(starts[index+1] if index<6 else 10**30) and family(k.name)=='dense_projections']
    assert Counter(key(k) for k in selected)==Counter({k:len(v) for k,v in template.items()})
    consumed: Counter[tuple[str,tuple[int,int,int],tuple[int,int,int]]]=Counter()
    for k in selected:
        role=template[key(k)][consumed[key(k)]];consumed[key(k)]+=1
        role_times[role]+=k.end-k.start;role_counts[role]+=1
results['sglang_graph_dense_roles']={role:{'ms_per_step':v/7/1e6,'kernels_per_step':role_counts[role]/7} for role,v in role_times.items()}
results['sglang_role_mapping']={'unique_keys':len(template),'kernels_per_step':sum(map(len,template.values())),'all_seven_graph_key_counts_match':True}
c.close()

# Rust graph projections are ordered; validate every role against its output grid.
c=sqlite3.connect(OUT/'rust-graph.sqlite');kernels=read_kernels(c);c.close()
groups=[];current=[]
for k in kernels:
    current.append(k)
    if k.name=='argmax_final_entry':groups.append(current);current=[]
expected: list[tuple[str,int,int]]=[]
for layer in range(40):
    if layer%4!=3:
        expected += [('linear_attn.in_proj_qkv',8192,2048),('linear_attn.in_proj_a',32,2048),('linear_attn.in_proj_b',32,2048),('linear_attn.in_proj_z',4096,2048),('linear_attn.out_proj',2048,4096)]
    else:
        expected += [('q_proj',8192,2048),('k_proj',512,2048),('v_proj',512,2048),('o_proj',2048,4096)]
    expected += [('mlp.gate',256,2048),('mlp.shared_expert.gate_proj',512,2048),('mlp.shared_expert.up_proj',512,2048),('mlp.shared_expert.down_proj',2048,512),('mlp.shared_expert_gate',1,2048)]
expected += [('lm_head',248320,2048)]
role_times=Counter();role_counts=Counter();shape_by_role={}
for group in groups[1:]:
    dense=[k for k in group if k.name=='linear_entry']
    assert len(dense)==len(expected)==391
    for k,(role,n,reduction) in zip(dense,expected):
        assert k.grid==(1,(n+63)//64,1),(role,k.grid)
        role_times[role]+=k.end-k.start;role_counts[role]+=1;shape_by_role[role]={'N':n,'K':reduction}
results['rust_graph_dense_roles']={role:{'ms_per_step':v/7/1e6,'kernels_per_step':role_counts[role]/7,**shape_by_role[role]} for role,v in role_times.items()}

controls=[]
for variant in ['default','serial']:
    for repetition in range(3):
        path=OUT/f'overlap-r{repetition}-{variant}.jsonl'
        if not path.exists():continue
        records=[json.loads(line) for line in path.read_text().splitlines()]
        assert len(records)==1
        r=records[0];assert r['cuda_graph'] and r['input_checksum']==5125296 and r['dtype']=='bfloat16'
        assert len(r['samples'])==3
        controls.append({'variant':variant,'repetition':repetition,'decode_ms':median(sample['median_decode_latency'] for sample in r['samples'])*1000})
results['native_overlap_control']={'processes':controls,'medians':{variant:median(r['decode_ms'] for r in controls if r['variant']==variant) for variant in ['default','serial']}}
(OUT/'profile-analysis.json').write_text(json.dumps(results,indent=2)+'\n')
for variant in ['rust-graph','sglang-graph']:
    print(variant,results[variant]['mean'])
    print(results[variant]['families'])
print('Rust launch-preparation median ms',median(r['post_upload_launch_preparation_ms'] for r in results['rust_replay_cpu']))
print('Rust host-validation median ms',median(results['rust_validation_ms']))
print('Eager role mappings',[(name,round(r['ms_per_step'],4),r['kernels_per_step']) for name,r in results['sglang_eager_dense_roles'].items()])
