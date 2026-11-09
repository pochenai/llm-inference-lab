"""Current graph attribution; profile durations are not native latencies."""
from pathlib import Path
from collections import Counter,defaultdict
import json,sqlite3,statistics
out=Path(__file__).absolute().parent
c=sqlite3.connect(out/'rust-graph.sqlite')
k=[dict(zip(['start','end','name','stream','gx','gy','gz'],r)) for r in c.execute('SELECT k.start,k.end,s.value,k.streamId,k.gridX,k.gridY,k.gridZ FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start')]
groups=[];current=[]
for row in k:
 current.append(row)
 if row['name']=='argmax_final_entry':groups.append(current);current=[]
assert len(groups)==8 and not current
steps=groups[1:]
ranges=[(n,a,b,t) for n,a,b,t in c.execute('SELECT COALESCE(n.text,s.value),n.start,n.end,n.globalTid FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId') if n and n.startswith('qwen3_5.')]
api=list(c.execute('SELECT a.start,a.end,s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME a JOIN StringIds s ON s.id=a.nameId ORDER BY a.start'))
replay=[]
for n,a,b,t in sorted(ranges,key=lambda r:r[1]):
 if n!='qwen3_5.decode.graph_replay':continue
 copies=[(x,y) for x,y,name in api if a<=x<b and name.startswith('cuMemcpyHtoDAsync')]
 launches=[(x,y) for x,y,name in api if a<=x<b and name=='cuGraphLaunch']
 assert len(copies)==2 and len(launches)==1
 replay.append({'metadata_preparation_ms':(copies[0][0]-a)/1e6,'post_upload_prelaunch_ms':(launches[0][0]-copies[-1][1])/1e6,'driver_launch_ms':(launches[0][1]-launches[0][0])/1e6})
validations=sorted([(a,b) for n,a,b,t in ranges if n=='qwen3_5.host_validation'])
tails=[]
for group,(a,b) in zip(steps,validations[1:]):
 copies=c.execute('SELECT end FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind=2 AND start>=? AND end<=?',(group[-1]['end'],a)).fetchall()
 assert len(copies)==2
 tails.append((a-max(r[0] for r in copies))/1e6)
counts=Counter();times=Counter()
for step in steps:
 for row in step:counts[row['name']]+=1;times[row['name']]+=row['end']-row['start']
roles=[]
kinds=json.loads(Path('/root/autodl-tmp/now/models/Qwen3.5-35B-A3B/config.json').read_text())['text_config']['layer_types']
for kind in kinds:
 roles+=([('gdn_qkv',128),('gdn_ab',1),('gdn_z',64),('mixer_out',32)] if kind=='linear_attention' else [('gqa_q',128),('gqa_k',8),('gqa_v',8),('mixer_out',32)])
 roles += [('router',64),('shared_gate_up',128),('shared_down',32),('shared_scalar',1)]
roles += [('lm_head',3880)]
dense={'linear_entry','linear_vector_entry','shared_gate_up_vector_entry'}
role_ns=Counter();role_counts=Counter()
for step in steps:
 projections=[r for r in step if r['name'] in dense]
 assert len(projections)==len(roles)==321
 for row,(role,grid) in zip(projections,roles):
  assert row['gy']==grid,(role,row)
  role_ns[role]+=row['end']-row['start'];role_counts[role]+=1
result={'steps':len(steps),'kernel_counts':[len(s) for s in steps],'kernel_sum_ms':[sum(r['end']-r['start'] for r in s)/1e6 for s in steps],'kernel_span_ms':[(s[-1]['end']-s[0]['start'])/1e6 for s in steps],'kernels':[{'name':n,'count_per_step':counts[n]/7,'ms_per_step':v/7/1e6} for n,v in times.most_common()],'dense_roles':{n:{'count_per_step':role_counts[n]/7,'ms_per_step':v/7/1e6} for n,v in role_ns.items()},'replay_cpu':replay,'replay_cpu_median':{n:statistics.median(r[n] for r in replay) for n in replay[0]},'post_readback_tail_ms':tails,'post_readback_tail_median_ms':statistics.median(tails),'validation_ms':[(b-a)/1e6 for a,b in validations[1:]],'validation_median_ms':statistics.median((b-a)/1e6 for a,b in validations[1:])}
(out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({n:v for n,v in result.items() if n in ['kernel_counts','kernel_sum_ms','kernel_span_ms','dense_roles','replay_cpu_median','post_readback_tail_median_ms','validation_median_ms']},indent=2))
