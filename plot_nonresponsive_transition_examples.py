"""Zoom NR-to-NR and NR-to-unlabeled examples from nearest-snapped labels."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUTPUT_DIR=Path('plots/arousal_transition_exploration')
EXAMPLES=(('NR_to_NR','AAN_07','run01',2594),('NR_to_NR','AAN_09','run03',1925),
          ('NR_to_unlabeled','AAN_03','run01',456),('NR_to_unlabeled','AAN_08','run07',338))


def main():
    root=Path('plots/GRU/non_responsiveness/T_min=20,BP=nearest')
    nr=pd.concat([pd.read_csv(root/name) for name in
        ('non_responsive_intervals.csv','non_responsive_end_intervals.csv','non_responsive_whole_runs.csv')])
    r=pd.read_csv('plots/GRU/responsiveness/W=30,S=1,N_min=8,G_max=4,BP=nearest/responsive_intervals.csv')
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    for category,participant,run,start in EXAMPLES:
        n=nr.loc[nr.participant.eq(participant)&nr.run.eq(run)].sort_values('start_TR')
        responsive=r.loc[r.participant.eq(participant)&r.run.eq(run)]
        a=n.loc[n.start_TR.eq(start)].iloc[0];end=int(a.end_TR)
        assert not ((responsive.start_TR<=end)&(responsive.end_TR>end)).any()
        next_nr=n.loc[n.start_TR.ge(end)].iloc[0]
        next_start,next_end=int(next_nr.start_TR),int(next_nr.end_TR)
        assert (next_start==end)==(category=='NR_to_NR')
        if category=='NR_to_unlabeled':
            assert not ((responsive.start_TR<next_start)&(responsive.end_TR>end)).any()
        left=max(0,start-40);right=next_end+10
        fig,ax=plt.subplots(figsize=(12,4),layout='constrained')
        for frame,color,alpha in ((n,'#E15759',.17),(responsive,'#59A14F',.2)):
            for row in frame.loc[frame.start_TR.lt(right)&frame.end_TR.gt(left)].itertuples():
                ax.axvspan(row.start_TR,row.end_TR,color=color,alpha=alpha,linewidth=0)
        raw=np.load(Path('raw_data')/participant/run/'bps.npy')
        presses=np.unique(np.floor(raw/.98+.5).astype(int))
        assert end in presses and next_start in presses
        presses=presses[(presses>=left)&(presses<=right)]
        ax.vlines(presses,.42,.58,color='#222222',linewidth=1.8,zorder=3)
        ax.axvline(end,color='red',linestyle='--',linewidth=2,zorder=4)
        ax.text((start+end)/2,.87,f'NR = {int(a.duration_TRs)} TRs',ha='center',va='center',fontsize=18,color='#8c2528')
        ax.text((next_start+next_end)/2,.87,f'NR = {int(next_nr.duration_TRs)} TRs',ha='center',va='center',fontsize=18,color='#8c2528')
        if category=='NR_to_unlabeled':
            ax.text((end+next_start)/2,.72,f'Unlabeled\n{next_start-end} TRs',ha='center',va='center',fontsize=15,color='.3')
        ax.set_title(f'{participant} {run}',fontsize=24,pad=16)
        ax.set_xlim(left,right);ax.set_ylim(0,1.1);ax.set_yticks([])
        ax.set_xlabel('TR',fontsize=19);ax.tick_params(axis='x',labelsize=16,width=1.8,length=7)
        ax.grid(axis='x',alpha=.15);ax.spines[['top','right','left']].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.8)
        handles=[Patch(facecolor='#E15759',alpha=.17,label='Non-responsive interval'),
                 Patch(facecolor='white',edgecolor='.75',label='Unlabeled'),
                 Line2D([],[],marker='|',linestyle='none',color='#222222',markersize=13,label='Nearest-snapped press')]
        fig.legend(handles=handles,loc='outside lower center',ncol=3,fontsize=12,frameon=False)
        path=OUTPUT_DIR/f'{category},{participant}_{run},start={start}.png'
        fig.savefig(path,dpi=200);plt.close(fig)
        print(f'Saved {path}: NR [{start},{end}) -> {category}; next NR [{next_start},{next_end})')


if __name__=='__main__':main()
