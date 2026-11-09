import json,statistics
from pathlib import Path
out=Path('rust/profiling/qwen3_5/cuda-graph-20261108');processes=[];summaries={};ids=[]
for engine in ['baseline','candidate','sglang-graph']:
 values=[]
 for rep in range(3):
  f=out/(f'r{rep}-{engine}.jsonl' if engine=='sglang-graph' else f'r{rep}-{engine}.stdout');records=[json.loads(l) for l in f.read_text().splitlines() if l.startswith('{')]
  if engine=='sglang-graph':
   assert len(records)==1;r=records[0];assert r['dtype']=='bfloat16' and r['cuda_graph'] and r['input_len']==32 and r['output_len']==8 and r['batch_size']==1 and r['input_checksum']==5125296
   ss=r['samples'];assert len(ss)==3;prefill=statistics.mean(s['prefill_latency'] for s in ss);decode=statistics.median(s['median_decode_latency'] for s in ss);total=statistics.mean(s['total_latency'] for s in ss)
  else:
   loaded=next(r for r in records if r.get('event')=='loaded');assert loaded['dtype']=='bf16' and loaded['weight_bytes']==69321574912 and not loaded['gpu_feedback'] and loaded['logits_validation'] and loaded['cuda_graph']==(engine=='candidate')
   ss=[r for r in records if r.get('event')=='sample' and not r['warmup']];assert len(ss)==3
   for s in ss:
    assert s['kv_released'] and s['forward_count']==8 and len(s['generated_ids'])==8 and len(s['decode_step_seconds'])==7;ids.append(s['generated_ids'])
    if engine=='candidate':assert s['workspace_allocations']==0 and s['workspace']['in_use_workspaces']==1 and s['cuda_graph']['captures']==1 and s['cuda_graph']['cached_graphs']==1 and s['cuda_graph']['replays']==(s['iteration']+1)*7
   prefill=statistics.mean(s['prefill_seconds'] for s in ss);decode=statistics.median(statistics.median(s['decode_step_seconds']) for s in ss);total=statistics.mean(s['prefill_seconds']+sum(s['decode_step_seconds']) for s in ss)
  value={'engine':engine,'repetition':rep,'prefill_ms':prefill*1000,'decode_ms':decode*1000,'total_ms':total*1000};values.append(value);processes.append(value)
 summaries[engine]={key:statistics.median(v[key] for v in values) for key in ['prefill_ms','decode_ms','total_ms']}
assert all(sequence==ids[0] for sequence in ids)
old,new,sg=[summaries[e] for e in ['baseline','candidate','sglang-graph']]
result={'engines':summaries,'prefill_reduction_percent':(1-new['prefill_ms']/old['prefill_ms'])*100,'decode_reduction_percent':(1-new['decode_ms']/old['decode_ms'])*100,'decode_speedup':old['decode_ms']/new['decode_ms'],'graph_decode_gap':new['decode_ms']/sg['decode_ms'],'total_reduction_percent':(1-new['total_ms']/old['total_ms'])*100,'measured_scratch_allocations':0,'all_rust_ids_equal':True}
(out/'processes.json').write_text(json.dumps(processes,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
for row in processes:print(row)
print(json.dumps(result,indent=2))
