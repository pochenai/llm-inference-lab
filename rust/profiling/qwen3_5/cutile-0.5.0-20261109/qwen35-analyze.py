import json,statistics
from pathlib import Path
out=Path('rust/profiling/qwen3_5/cutile-0.5.0-20261109');all_processes=[];generated=[];summaries={}
for engine in ['rust','sglang','sglang-graph']:
 processes=[]
 for rep in range(3):
  f=out/(f'r{rep}-rust.stdout' if engine=='rust' else f'r{rep}-{engine}.jsonl');records=[json.loads(l) for l in f.read_text().splitlines() if l.strip()]
  if engine=='rust':
   loaded=next(r for r in records if r['event']=='loaded');assert loaded['weight_bytes']==69321574912 and loaded['batch_size']==1 and loaded['dtype']=='bf16'
   samples=[r for r in records if r['event']=='sample' and not r['warmup']];assert len(samples)==3
   for sample in samples:
    assert sample['kv_released'] and sample['forward_count']==8 and len(sample['decode_step_seconds'])==7 and len(sample['generated_ids'])==8
    generated.append(sample['generated_ids'])
   prefill=statistics.mean(s['prefill_seconds'] for s in samples);decode=statistics.median(statistics.median(s['decode_step_seconds']) for s in samples);total=statistics.mean(s['prefill_seconds']+sum(s['decode_step_seconds']) for s in samples)
  else:
   assert len(records)==1;r=records[0];assert r['sglang_version']=='0.5.9' and r['dtype']=='bfloat16' and r['batch_size']==1 and r['input_len']==32 and r['output_len']==8 and r['input_checksum']==5125296 and r['cuda_graph']==(engine=='sglang-graph')
   samples=r['samples'];assert len(samples)==3
   prefill=statistics.mean(s['prefill_latency'] for s in samples);decode=statistics.median(s['median_decode_latency'] for s in samples);total=statistics.mean(s['total_latency'] for s in samples)
  row={'engine':engine,'repetition':rep,'prefill_ms':prefill*1000,'decode_ms':decode*1000,'total_ms':total*1000};processes.append(row);all_processes.append(row)
 summaries[engine]={key:statistics.median(p[key] for p in processes) for key in ['prefill_ms','decode_ms','total_ms']}
assert all(ids==generated[0] for ids in generated)
(out/'processes.json').write_text(json.dumps(all_processes,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
lines=['# Qwen3.5-35B-A3B: initial single-GPU latency baseline','','RTX 6000D, CUDA 13.2, cuTile 0.5.0, SGLang 0.5.9, CPU affinity 24–31 on NUMA 0. Native BF16 checkpoint; SGLang uses explicit Triton attention and FP32 Mamba SSM state. B1, 32 repeated token-9707 inputs, exactly eight outputs, fresh session state. Two warmups and three measured samples per process; three independent processes/engine. Eager engine order alternates. SGLang graph references run separately after eager samples. No profiler, compilation or GPU tests overlap native timing. CPU/GPU clocks and unrelated host load are not controlled.','','Each cell is the median of three process aggregates: mean prefill/total, median of sample median decode steps.','','| Engine | Prefill ms | Decode ms/step | Total for eight outputs ms |','| --- | ---: | ---: | ---: |']
for engine,value in summaries.items():lines.append(f"| {engine} | {value['prefill_ms']:.3f} | {value['decode_ms']:.3f} | {value['total_ms']:.3f} |")
lines+=['','## Measurement boundaries','','- Rust directly calls the pooled CausalModel forward, with GPU argmax, full-logit D2H/finite-value validation and completed host token feedback. Lazy recurrent-state initialization is included in prefill. Weights, JIT and pool backing storage stay warm. No Scheduler, HTTP, tokenization or text decode is timed.','- SGLang uses the existing static-batch adapter, GPU sampling and explicit Triton full attention/GDN. Graph accelerates decode; prefill remains eager. Vision weights are resident in its conditional-generation wrapper, but these samples contain no images or encoder work. Pool capacities differ, so this is not an allocator-efficiency comparison.','- Rust has no Qwen3.5 graph or GPU-feedback path yet. Its eager boundary differs from the graph reference. The observed decode gap is an optimization target, not an isolated kernel or quality ranking.','- SGLang reports no RTX 6000D-specific MoE tuning file and uses its default MoE config; this is the stock installed baseline, not a claim about the fastest tuned SGLang.','- All nine measured Rust samples reproduce the same complete generated IDs, run exactly eight forwards, and release session resources. SGLang samples retain input checksums and timing records but not generated IDs; cross-engine history/output equivalence is not established.','',f"Rust eager / SGLang graph decode ratio: **{summaries['rust']['decode_ms']/summaries['sglang-graph']['decode_ms']:.2f}x**. Full-cohort ratio: **{summaries['rust']['total_ms']/summaries['sglang-graph']['total_ms']:.2f}x**.",'','## Independent process observations','','| Engine | Prefill ms by process | Decode ms by process |','| --- | --- | --- |']
for engine in summaries:
 ps=[p for p in all_processes if p['engine']==engine];lines.append('| '+engine+' | '+', '.join(f"{p['prefill_ms']:.3f}" for p in ps)+' | '+', '.join(f"{p['decode_ms']:.3f}" for p in ps)+' |')
lines+=['','Process variation is material, especially in eager Python submission; no confidence-bound improvement claim is made. Frozen native Rust source/binary, raw samples, commands and metadata accompany this report. The separate profiling executable adds only profiler-range control; its source/binary and trace are retained separately. Failed language-only and automatic-attention startup attempts are excluded and preserved in failed-*.stderr.','']
(out/'RESULTS.md').write_text('\n'.join(lines));print('\n'.join(lines[:14]))
