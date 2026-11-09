from pathlib import Path
import collections,csv,json,sqlite3
out=Path('rust/profiling/qwen3_5/workspace-reuse-20261109');summary={}
for variant in ['baseline','candidate']:
 c=sqlite3.connect(out/f'profile-{variant}.sqlite');kernels=c.execute('SELECT k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start').fetchall();steps=[];current=[]
 for start,end,name in kernels:
  current.append((name,end-start))
  if name=='argmax_final_entry':steps.append(current);current=[]
 assert len(steps)==8 and not current
 phases={}
 for label,groups in [('prefill',steps[:1]),('decode',steps[1:])]:
  times=collections.Counter();counts=collections.Counter()
  for group in groups:
   for name,duration in group:times[name]+=duration;counts[name]+=1
  total=sum(times.values());phases[label]={'kernel_ms_per_forward':total/len(groups)/1e6,'kernel_instances':sum(counts.values()),'kernels':[{'name':name,'count':counts[name],'total_ms':duration/1e6,'share_percent':duration/total*100} for name,duration in times.most_common()]}
 blocks=(out/f'profile-{variant}-stats.csv').read_text().split('Time (%),Total Time (ns),');api={}
 for line in blocks[1].splitlines()[1:]:
  if not line.strip():break
  row=next(csv.reader([line]));api[row[-1]]={'calls':int(row[2]),'total_ms':int(row[1])/1e6}
 summary[variant]={'api':api,'phases':phases}
 if variant=='candidate':
  ranges=c.execute('SELECT COALESCE(n.text,s.value),n.start,n.end,n.globalTid,n.endGlobalTid FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId').fetchall();ranges=[r for r in ranges if r[0] and r[0].startswith('qwen3_5.')]
  assert all(end is not None and end>=start and (end_tid is None or end_tid==tid) for _,start,end,tid,end_tid in ranges)
  counts=collections.Counter(r[0] for r in ranges);assert counts['qwen3_5.prefill.submit']==1 and counts['qwen3_5.decode.submit']==7 and counts['qwen3_5.workspace.checkout']==8 and counts['qwen3_5.workspace.allocate']==0
  assert counts['qwen3_5.state.allocate']==1 and counts['qwen3_5.host_validation']==8
  for layer in range(40):assert counts[f'qwen3_5.layer.{layer}']==8
  assert counts['qwen3_5.mixer.delta']==240 and counts['qwen3_5.mixer.gqa']==80 and counts['qwen3_5.moe.expert.gate_up']==320 and counts['qwen3_5.moe.expert.down']==320
  summary[variant]['nvtx_counts']=dict(sorted(counts.items()))
(out/'profile-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for variant,s in summary.items():print(variant,{k:s['api'][k]['calls'] for k in ['cuLaunchKernel','cuMemAllocAsync','cuMemFreeAsync','cuEventRecord']}, {k:round(p['kernel_ms_per_forward'],3) for k,p in s['phases'].items()})
print('NVTX counts validated')
