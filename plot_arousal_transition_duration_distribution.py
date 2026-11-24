"""Plot NR and R duration distributions among direct nearest-cohort NR-to-R transitions."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR=Path('plots/arousal_transition_exploration')


def plot_distribution(rows, interval_type):
    column = 'nonresponsive_duration_TRs' if interval_type == 'NR' else 'responsive_duration_TRs'
    durations=np.array([int(row[column]) for row in rows])
    edges=np.r_[np.arange(20,91,10),np.inf]
    counts,_=np.histogram(durations,bins=edges)
    assert len(durations)==58 and counts.sum()==len(durations)
    labels=[f'[{a}, {a+10})' for a in range(20,90,10)]+['90+']
    fig,ax=plt.subplots(figsize=(max(14,len(labels)*.55),7),layout='constrained')
    positions=np.arange(len(labels))
    bars=ax.bar(positions,counts,width=.7,color=('pink' if interval_type == 'NR' else '#9ADF8C'),alpha=1,edgecolor='black',linewidth=1.8,zorder=3)
    for bar,count in zip(bars,counts):
        if count:
            ax.text(bar.get_x()+bar.get_width()/2,count+.15,str(count),ha='center',va='bottom',fontsize=21)
    ax.set_xticks(positions,labels,rotation=0)
    ax.set_xlabel(('Non-responsive' if interval_type == 'NR' else 'Responsive') + ' interval duration (TRs)',fontsize=27,labelpad=18)
    ax.set_ylabel('Number of intervals',fontsize=27,labelpad=18)
    title = 'NR intervals ending in R interval (n=58)' if interval_type == 'NR' else 'R intervals following NR interval (n=58)'
    ax.set_title(title,fontsize=30,pad=22)
    ax.set_ylim(0,max(counts)*1.3)
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis='both',labelsize=23,width=2,length=7)
    ax.tick_params(axis='x',labelsize=20)
    ax.grid(axis='y',alpha=.3,linewidth=1.5);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_linewidth(2)
    path=OUTPUT_DIR/f'NR_to_R,duration_distribution_{interval_type}.png'
    fig.savefig(path,dpi=200);plt.close(fig)
    print(f'Saved {path}')
    for label,count in zip(labels,counts):print(f'{label}: {count}')


def main():
    with (OUTPUT_DIR/'NR_to_R_intervals.csv').open() as handle:
        rows=list(csv.DictReader(handle))
    for interval_type in ('NR', 'R'):
        plot_distribution(rows, interval_type)


if __name__=='__main__':main()
