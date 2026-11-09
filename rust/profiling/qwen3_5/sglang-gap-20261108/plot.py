"""Standalone timeline of the last profiled decode, aligned to its first GPU kernel."""
from pathlib import Path
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

out=Path('rust/profiling/qwen3_5/sglang-gap-20261108')
colors={'dense':'#247ba0','expert':'#f28e2b','attention':'#59a14f','other':'#b6bbc3'}


def color(name: str) -> str:
    if name=='linear_entry' or any(x in name for x in ['gemvx::','splitKreduce','cutlass::Kernel']):return colors['dense']
    if name in ['expert_entry','fused_moe_kernel']:return colors['expert']
    if name in ['gqa_graph_entry','_fwd_grouped_kernel_stage1','_fwd_kernel_stage2']:return colors['attention']
    return colors['other']


fig,ax=plt.subplots(figsize=(13,5))
for variant in ['rust-graph','sglang-graph']:
    c=sqlite3.connect(out/(variant+'.sqlite'))
    kernels=c.execute('SELECT k.start,k.end,s.value,k.streamId FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start').fetchall()
    if variant=='rust-graph':
        groups=[];current=[]
        for k in kernels:
            current.append(k)
            if k[2]=='argmax_final_entry':groups.append(current);current=[]
        chosen=groups[-1];origin=chosen[0][0]
        for start,end,name,stream in chosen:ax.broken_barh([((start-origin)/1e6,(end-start)/1e6)],(4.5,.55),facecolors=color(name),linewidth=0)
        rs=c.execute("SELECT n.start,n.end FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId WHERE COALESCE(n.text,s.value)='qwen3_5.decode.graph_replay' ORDER BY n.start").fetchall();start,end=rs[-1]
        launch=c.execute("SELECT a.start,a.end FROM CUPTI_ACTIVITY_KIND_RUNTIME a JOIN StringIds s ON s.id=a.nameId WHERE s.value='cuGraphLaunch' AND a.start>=? AND a.start<?",(start,end)).fetchone()
        ax.broken_barh([((start-origin)/1e6,(launch[0]-start)/1e6)],(5.5,.55),facecolors='#a866b5',linewidth=0)
        ax.broken_barh([((launch[0]-origin)/1e6,(launch[1]-launch[0])/1e6)],(5.5,.55),facecolors='#6b3b83',linewidth=0)
        val=c.execute("SELECT n.start,n.end FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId WHERE COALESCE(n.text,s.value)='qwen3_5.host_validation' ORDER BY n.start").fetchall()[-1]
        ax.broken_barh([((val[0]-origin)/1e6,(val[1]-val[0])/1e6)],(5.5,.55),facecolors='#e15759',linewidth=0)
        ax.annotate('CPU replay preparation',xy=((launch[0]-origin)/1e6*.6,5.8),xytext=(1.6,6.2),arrowprops={'arrowstyle':'->','color':'#555'},fontsize=10)
    else:
        start=c.execute("SELECT MAX(n.start) FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId WHERE COALESCE(n.text,s.value)='sglang.decode'").fetchone()[0]
        chosen=[k for k in kernels if k[0]>=start];origin=chosen[0][0]
        streams=sorted(set(k[3] for k in chosen))
        assert len(streams)==3
        positions={streams[0]:.5,streams[1]:2.5,streams[2]:1.5}
        for start,end,name,stream in chosen:ax.broken_barh([((start-origin)/1e6,(end-start)/1e6)],(positions[stream],.55),facecolors=color(name),linewidth=0)
    c.close()
ax.set_yticks([5.775,4.775,2.775,1.775,.775],['Rust CPU (selected ranges)','Rust GPU','SGLang main GPU stream','SGLang alternate GPU stream','SGLang GPU preparation / sampling'])
ax.set_xlim(-.1,11.5);ax.set_ylim(0,6.7)
ax.set_xlabel('Milliseconds since first GPU kernel of each engine\'s last decode')
ax.set_title('Profiled BF16 B1 decode: kernel execution and overlap\nDifferent engine origins are aligned; this is not the native latency table',fontsize=12)
ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
ax.legend(handles=[Patch(color=v,label=k) for k,v in colors.items()]+[Patch(color='#a866b5',label='CPU replay prep'),Patch(color='#6b3b83',label='driver graph launch'),Patch(color='#e15759',label='CPU logit validation')],loc='upper center',bbox_to_anchor=(.48,-.16),ncol=4,frameon=False,fontsize=9)
fig.tight_layout();fig.savefig(out/'decode-timeline.svg',bbox_inches='tight');fig.savefig(out/'decode-timeline.png',dpi=160,bbox_inches='tight')
