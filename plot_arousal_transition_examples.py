"""Zoom four nearest-cohort NR-to-R transitions using the run-label plot inputs."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

SEGMENT_DIR=Path('plots/GRU/segments/R(W=30,S=1,N_min=8,G_max=4):NR(T_min=20):Segment(TR=40,SW=40,BP=nearest)')
OUTPUT_DIR=Path('plots/arousal_transition_exploration')
EXAMPLES=(('AAN_05','run03',1806),('AAN_07','run01',2765),
          ('AAN_05','run02',1028),('AAN_12','run04',2400))
COLORS={'responsive':'#59A14F','non-responsive':'#E15759'}


def main():
    transitions=pd.read_csv(OUTPUT_DIR/'NR_to_R_intervals.csv')
    responsive=pd.read_csv('plots/GRU/responsiveness/W=30,S=1,N_min=8,G_max=4,BP=nearest/responsive_intervals.csv')
    nr_root=Path('plots/GRU/non_responsiveness/T_min=20,BP=nearest')
    nonresponsive=pd.concat([pd.read_csv(nr_root/name) for name in
        ('non_responsive_intervals.csv','non_responsive_end_intervals.csv','non_responsive_whole_runs.csv')])
    for participant,run,start in EXAMPLES:
        source=SEGMENT_DIR/'labels'/f'{participant}_{run}_labels.png'
        if not source.exists():raise FileNotFoundError(source)
        match=transitions.loc[transitions.participant.eq(participant)&transitions.run.eq(run)&transitions.nonresponsive_start_TR.eq(start)]
        if len(match)!=1:raise ValueError('Example must match exactly one transition')
        t=match.iloc[0];end=int(t.nonresponsive_end_TR_exclusive)
        r_start,r_end=int(t.responsive_start_TR),int(t.responsive_end_TR_exclusive)
        left=max(0,start-40);right=r_end+10
        fig,ax=plt.subplots(figsize=(12,4),layout='constrained')
        for intervals,label,alpha in ((responsive,'responsive',.2),(nonresponsive,'non-responsive',.17)):
            selected=intervals.loc[intervals.participant.eq(participant)&intervals.run.eq(run)&intervals.start_TR.lt(right)&intervals.end_TR.gt(left)]
            for row in selected.itertuples():ax.axvspan(row.start_TR,row.end_TR,color=COLORS[label],alpha=alpha,linewidth=0)
        raw=np.asarray(np.load(Path('raw_data')/participant/run/'bps.npy'),dtype=float)
        presses=np.unique(np.floor(raw/.98+.5).astype(int));presses=presses[(presses>=left)&(presses<=right)]
        ax.vlines(presses,.42,.58,color='#222222',linewidth=1.8,zorder=3)
        ax.axvline(end,color='red',linestyle='--',linewidth=2,zorder=4)
        ax.text((start+end)/2,.87,f'NR = {int(t.nonresponsive_duration_TRs)} TRs',
                ha='center',va='center',fontsize=19,color='#8c2528')
        ax.text((r_start+r_end)/2,.87,f'R = {int(t.responsive_duration_TRs)} TRs',
                ha='center',va='center',fontsize=19,color='#286c2a')
        ax.set_title(f'{participant} {run}',fontsize=24,pad=16)
        ax.set_xlim(left,right);ax.set_ylim(0,1.1);ax.set_yticks([])
        ax.set_xlabel('TR',fontsize=19);ax.tick_params(axis='x',labelsize=16,width=1.8,length=7)
        ax.grid(axis='x',alpha=.15);ax.spines[['top','right','left']].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.8)
        handles=[Patch(facecolor=COLORS['non-responsive'],alpha=.17,label='Non-responsive interval'),
                 Patch(facecolor=COLORS['responsive'],alpha=.2,label='Responsive interval'),
                 Line2D([],[],marker='|',linestyle='none',color='#222222',markersize=13,label='Nearest-snapped press')]
        fig.legend(handles=handles,loc='outside lower center',ncol=3,fontsize=12,frameon=False)
        output=OUTPUT_DIR/f'NR_to_R,{participant}_{run},start={start}.png'
        fig.savefig(output,dpi=200);plt.close(fig);print(f'Saved {output}')


if __name__=='__main__':main()
