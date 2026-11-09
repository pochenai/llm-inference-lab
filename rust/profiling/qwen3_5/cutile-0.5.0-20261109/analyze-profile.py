import collections,json,sqlite3
from pathlib import Path
out=Path('rust/profiling/qwen3_5/cutile-0.5.0-20261109');connection=sqlite3.connect(out/'rust-profile.sqlite')
kernels=connection.execute('SELECT k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start').fetchall();steps=[];current=[]
for start,end,name in kernels:
 current.append((name,end-start))
 if name=='argmax_final_entry':steps.append(current);current=[]
assert len(steps)==8 and not current,(len(steps),len(current))
phases={}
for label,groups in [('prefill',steps[:1]),('decode',steps[1:])]:
 totals=collections.Counter();counts=collections.Counter()
 for group in groups:
  for name,duration in group:totals[name]+=duration;counts[name]+=1
 total=sum(totals.values());phases[label]={'forwards':len(groups),'kernel_instances':sum(counts.values()),'total_kernel_ms':total/1e6,'kernel_ms_per_forward':total/len(groups)/1e6,'kernels':[{'name':name,'total_ms':duration/1e6,'share_percent':duration/total*100,'count':counts[name]} for name,duration in totals.most_common()]}
(out/'profile-summary.json').write_text(json.dumps(phases,indent=2)+'\n')
for label,phase in phases.items():print(label,phase['kernel_ms_per_forward'],phase['kernel_instances'],phase['kernels'][:4])
