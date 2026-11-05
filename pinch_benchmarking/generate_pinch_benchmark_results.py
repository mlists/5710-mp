#!/usr/bin/env python3
"""Create publication-ready figures and LaTeX tables for the AR chess pinch benchmark.

Default source: the verified numerical SUMMARY (6 Nov 2026) in
'Pinch_Gesture_Report_Handover_REDOWNLOAD(1).docx'. These data are embedded
below, allowing this script to run without the unavailable source CSVs/videos.

Optional: --session2-dir PATH reads selected evaluator CSVs and recomputes
figures wherever the schema can be recognised. A provenance log records which
values came from CSVs versus the handover. Use --strict-csv to require all
expected source tables to be parsed. Without the CSVs, do not describe the
figures as newly recomputed from raw data.

Usage:
    python generate_pinch_benchmark_results.py --out pinch_results
    python generate_pinch_benchmark_results.py --session2-dir /path/session02 --out pinch_results
    python generate_pinch_benchmark_results.py --videos /path/to/videos --out pinch_results
    python generate_pinch_benchmark_results.py --data-json pinch_benchmark_data.json --out pinch_results

Dependencies: numpy, matplotlib, pandas. OpenCV is optional for video contact sheets.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


SOURCE_DATA: dict[str, Any] = {
    'source': 'Pinch_Gesture_Report_Handover_REDOWNLOAD(1).docx, 6 November 2026',
    'session1': {
        'positive_test_conditions': 6,
        'positive_f1_each': 1.000,
        'negative_false_starts_per_min': 3.591,
        'median_abs_delay_ms': 113.0,
        'note': 'Session 1 figures read from prior plots, not independently re-aggregated CSVs.'
    },
    'session2': {
        'positive_clips': 13, 'negative_clips': 6, 'annotated_events': 295,
        'tp': 268, 'fp': 14, 'fn': 27, 'precision': 0.950,
        'recall': 0.908, 'f1': 0.929,
        'median_abs_delay_ms': 95.5,
        'negative_false_starts': 51, 'negative_duration_sec': 131.88,
        'replay_checked_clips': 19, 'replay_discrepancies': 0,
        'event_match_tolerance_sec': 0.5,
        'confirmation_frames': 3
    },
    'conditions': [
        {'family':'Viewpoint', 'condition':'Palm-facing',   'key':'view_palm', 'f1':1.000, 'hand_detection_pct':97.5},
        {'family':'Viewpoint', 'condition':'Back-of-hand',  'key':'view_back', 'f1':0.930, 'hand_detection_pct':89.7},
        {'family':'Working area', 'condition':'Near',       'key':'zone_near', 'f1':0.976, 'hand_detection_pct':97.4},
        {'family':'Working area', 'condition':'Far',        'key':'zone_far', 'f1':0.976, 'hand_detection_pct':88.1},
        {'family':'Working area', 'condition':'Left',       'key':'zone_left', 'f1':0.857, 'hand_detection_pct':73.1},
        {'family':'Working area', 'condition':'Right',      'key':'zone_right', 'f1':0.912, 'hand_detection_pct':95.7},
        {'family':'Back-of-hand yaw', 'condition':'45 deg yaw', 'key':'yaw_45', 'f1':0.966, 'hand_detection_pct':98.1},
        {'family':'Back-of-hand yaw', 'condition':'75 deg yaw', 'key':'yaw_75', 'f1':0.818, 'hand_detection_pct':96.7},
        {'family':'Interaction', 'condition':'Stationary hold', 'key':'interaction_hold', 'f1':0.976, 'hand_detection_pct':98.9},
        {'family':'Interaction', 'condition':'Pinch and drag', 'key':'interaction_drag', 'f1':0.889, 'hand_detection_pct':90.5},
        {'family':'Interaction', 'condition':'Virtual chess move', 'key':'interaction_chess', 'f1':0.824, 'hand_detection_pct':99.0},
    ],
    'negative_controls': [
        {'action':'Fast fist transitions',  'key':'fist_fast',  'n_clips':2, 'false_starts':25, 'duration_sec':23.22},
        {'action':'Slow fist transitions',  'key':'fist_slow',  'n_clips':2, 'false_starts':19, 'duration_sec':43.57},
        {'action':'Static fist',            'key':'fist_static','n_clips':1, 'false_starts':4,  'duration_sec':35.04},
        {'action':'Reach / hover',          'key':'reach_hover','n_clips':1, 'false_starts':3,  'duration_sec':30.05},
    ],
    'ablation': [
        {'frames':1, 'positive_f1':0.915, 'positive_recall':0.946, 'false_start_per_min':24.11, 'median_abs_delay_ms':32.0},
        {'frames':3, 'positive_f1':0.929, 'positive_recall':0.908, 'false_start_per_min':23.20, 'median_abs_delay_ms':95.5},
        {'frames':5, 'positive_f1':0.934, 'positive_recall':0.908, 'false_start_per_min':22.29, 'median_abs_delay_ms':164.0},
    ],
    'manually_inspected': [
        {'label':'75 deg yaw', 'clip_id':'orientation_back_yaw_75deg_01', 'fp':0, 'fn':8,
         'note':'Index/thumb tips occluded behind hand; 96.7% hand detection.'},
        {'label':'Board-left', 'clip_id':'board_zone_left_01', 'fp':0, 'fn':6,
         'note':'Partial hand exit/re-entry disrupts landmark extraction.'},
        {'label':'Virtual chess move', 'clip_id':'interaction_virtual_piece_move_01', 'fp':4, 'fn':2,
         'note':'Six audit entries traced to three premature START episodes.'}
    ],
    'virtual_move_episodes': [
        {'name':'Episode 1: false command', 'pred_start_sec':15.979, 'pred_release_sec':16.245, 'gt_start_sec':None},
        {'name':'Episode 2: early START', 'pred_start_sec':17.032, 'pred_release_sec':None, 'gt_start_sec':17.798},
        {'name':'Episode 3: early START', 'pred_start_sec':33.156, 'pred_release_sec':None, 'gt_start_sec':33.929}
    ]
}

# Clean, mostly neutral styles suited to a two-column IEEE report.
COLORS = {'Viewpoint':'#315b7c', 'Working area':'#4e83a9',
          'Back-of-hand yaw':'#bf7a40', 'Interaction':'#4b958e'}
DARK = '#25313d'
MID = '#687582'
LIGHT = '#dae2e9'
ACCENT = '#bd6b38'


def setup_style() -> None:
    plt.rcParams.update({
        'font.family':'DejaVu Sans', 'font.size':8.0, 'axes.labelsize':8.5,
        'xtick.labelsize':7.5, 'ytick.labelsize':7.5, 'legend.fontsize':7.0,
        'axes.edgecolor':'#a5b0b9', 'axes.linewidth':0.7,
        'axes.spines.top':False, 'axes.spines.right':False,
        'savefig.facecolor':'white', 'figure.facecolor':'white',
        'pdf.fonttype':42, 'ps.fonttype':42,
    })


def save(fig: plt.Figure, out: Path, basename: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    # PDF is vector and preferred for LaTeX; PNG is convenient for previews.
    fig.savefig(out / f'{basename}.pdf', bbox_inches='tight', pad_inches=0.06)
    fig.savefig(out / f'{basename}.png', dpi=350, bbox_inches='tight', pad_inches=0.06)
    plt.close(fig)


def calculate(data: dict) -> None:
    for r in data['negative_controls']:
        r['false_start_per_min'] = 60.0 * int(r['false_starts']) / float(r['duration_sec'])
    s = data['session2']
    s['negative_false_start_per_min'] = 60.0 * s['negative_false_starts'] / s['negative_duration_sec']


def validate(data: dict) -> None:
    s = data['session2']
    if s['tp'] + s['fn'] != s['annotated_events']:
        raise ValueError('Session 2: TP + FN does not equal number of annotated events.')
    computed_p = s['tp'] / (s['tp'] + s['fp'])
    computed_r = s['tp'] / (s['tp'] + s['fn'])
    computed_f1 = 2*computed_p*computed_r/(computed_p+computed_r)
    for label, expected in [('precision', computed_p),('recall', computed_r),('f1',computed_f1)]:
        if abs(float(s[label])-expected) > 0.002:
            raise ValueError(f"Session 2 {label} inconsistent: {s[label]} vs {expected:.4f}")
    negative = data['negative_controls']
    if sum(x['false_starts'] for x in negative) != s['negative_false_starts']:
        raise ValueError('Negative-control false START counts do not match total.')
    if abs(sum(x['duration_sec'] for x in negative)-s['negative_duration_sec']) > 0.03:
        raise ValueError('Negative-control durations do not match total.')
    if sorted(x['frames'] for x in data['ablation']) != [1,3,5]:
        raise ValueError('Expected 1/3/5 confirmation-frame ablation.')
    for row in data['conditions']:
        if not 0 <= row['f1'] <= 1 or not 0 <= row['hand_detection_pct'] <= 100:
            raise ValueError(f'Invalid condition value: {row}')


def plot_condition_f1(data: dict, out: Path) -> None:
    rows = data['conditions']
    y, labels, color = [], [], []
    group_positions: dict[str,list[float]] = defaultdict(list)
    cur = 0.0
    family = None
    for r in rows:
        if family is not None and r['family'] != family:
            cur += 0.48
        family = r['family']
        y.append(cur); labels.append(r['condition']); color.append(COLORS[family])
        group_positions[family].append(cur)
        cur += 1.0
    fig, ax = plt.subplots(figsize=(3.58, 4.08))
    ax.barh(y, [r['f1'] for r in rows], height=0.68, color=color, edgecolor='white', linewidth=0.3)
    ax.set_yticks(y, labels=labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.13); ax.set_xticks([0,.25,.5,.75,1])
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.set_xlabel('Event-level F1, Session 2 (3-frame baseline)')
    ax.tick_params(axis='y', length=0, pad=4)
    ax.grid(axis='x', color='#e1e6e9', linewidth=0.55)
    ax.set_axisbelow(True)
    for yy,r in zip(y, rows):
        ax.text(float(r['f1']) + 0.018, yy, f"{r['f1']:.3f}", ha='left', va='center',
                fontsize=7.0, color=DARK, weight='bold' if r['f1'] < 0.86 else 'normal')
    for name, ps in group_positions.items():
        if ps:
            ax.text(0.015, ps[0]-0.46, name.upper(), fontsize=6.35,
                    va='center', color=COLORS[name], weight='bold')
    ax.set_ylim(cur-0.28, -0.77)
    fig.subplots_adjust(left=.38, right=.965, top=.97, bottom=.125)
    save(fig, out, 'Fig01_F1_by_condition')


def plot_negative(data: dict, out: Path) -> None:
    rows = data['negative_controls']
    fig, ax = plt.subplots(figsize=(3.55, 2.46))
    y = np.arange(len(rows))
    rates = [r['false_start_per_min'] for r in rows]
    bars = ax.barh(y, rates, height=.60, color=['#b77446','#cf9a65','#7c9ba8','#7c9ba8'])
    ax.set_yticks(y, labels=[r['action'] for r in rows]); ax.invert_yaxis()
    ax.tick_params(axis='y', length=0, pad=4)
    ax.set_xlim(0, 99); ax.set_xticks([0,20,40,60,80])
    ax.set_xlabel('Unintended PINCH_START events / min')
    ax.grid(axis='x', color='#e1e6e9', linewidth=.6); ax.set_axisbelow(True)
    for bar, r in zip(bars, rows):
        ax.text(bar.get_width()+1.5, bar.get_y()+bar.get_height()/2,
                f"{r['false_start_per_min']:.2f}  ({r['false_starts']} / {r['duration_sec']:.2f} s)",
                va='center', ha='left', fontsize=6.9, color=DARK)
    fig.subplots_adjust(left=.37, right=.985, top=.94, bottom=.23)
    save(fig, out, 'Fig02_False_START_rates')


def plot_ablation(data: dict, out: Path) -> None:
    rows = sorted(data['ablation'], key=lambda r:r['frames'])
    x = np.array([r['frames'] for r in rows])
    fig, axs = plt.subplots(1, 3, figsize=(7.13, 2.10))
    axs[0].plot(x,[r['positive_f1'] for r in rows],marker='o',color=COLORS['Viewpoint'],label='F1',lw=1.65,ms=4)
    axs[0].plot(x,[r['positive_recall'] for r in rows],marker='s',color=COLORS['Interaction'],label='Recall',lw=1.5,ms=4)
    axs[0].set_ylim(.88,.98); axs[0].set_ylabel('Proportion');axs[0].legend(frameon=False,loc='lower left',ncol=2)
    axs[1].plot(x,[r['false_start_per_min'] for r in rows],marker='o',color=ACCENT,lw=1.65,ms=4)
    axs[1].set_ylim(20.5,25.5);axs[1].set_ylabel('False START / min')
    axs[2].plot(x,[r['median_abs_delay_ms'] for r in rows],marker='o',color=COLORS['Viewpoint'],lw=1.65,ms=4)
    axs[2].set_ylim(0,185); axs[2].set_ylabel('Median |delay| (ms)')
    for ax in axs:
        ax.set_xlabel('Confirmation frames');ax.set_xticks([1,3,5])
        ax.grid(alpha=.7,linewidth=.55,color='#e1e6e9');ax.set_axisbelow(True)
        ax.axvline(3,color='#a8b4be',lw=.9,ls=':')
    fig.subplots_adjust(wspace=.48,top=.95,bottom=.24,left=.07,right=.98)
    save(fig, out, 'Fig03_Confirmation_ablation')


def plot_tracking_vs_f1(data:dict,out:Path)->None:
    fig, ax=plt.subplots(figsize=(3.45,3.08))
    rows=data['conditions']
    for fam in COLORS:
        rr=[r for r in rows if r['family']==fam]
        ax.scatter([r['hand_detection_pct'] for r in rr], [r['f1']*100 for r in rr],
                   s=45,c=COLORS[fam],edgecolor='white',lw=.7, label=fam,zorder=3)
    anno={
        'yaw_75':('75° yaw',(-28,-16)),
        'zone_left':('Board left',(6,10)),
        'interaction_chess':('Chess move',(-66,14)),
        'zone_far':('Board far',(5,-12)),
    }
    for r in rows:
        if r['key'] in anno:
            label,off=anno[r['key']]
            ax.annotate(label,xy=(r['hand_detection_pct'],r['f1']*100),xytext=off,
                        textcoords='offset points',fontsize=6.9,color=DARK,
                        arrowprops=dict(arrowstyle='-',lw=.45,color=MID))
    ax.set_xlim(68,103);ax.set_ylim(78,104)
    ax.set_xlabel('Hand-detection rate (%)')
    ax.set_ylabel('Event-level F1 (%)')
    ax.grid(color='#e1e6e9',lw=.6);ax.set_axisbelow(True)
    ax.legend(loc='upper left',frameon=False,fontsize=6.1,handletextpad=.25,labelspacing=.3)
    fig.subplots_adjust(left=.18,right=.96,bottom=.17,top=.95)
    save(fig,out,'Fig04_Hand_tracking_vs_F1')


def plot_audited_errors(data:dict,out:Path)->None:
    rows=data['manually_inspected']
    y=np.arange(len(rows))
    fp=np.array([r['fp'] for r in rows]);fn=np.array([r['fn'] for r in rows])
    fig,ax=plt.subplots(figsize=(3.45,2.3))
    ax.barh(y,fn,height=.58,color=COLORS['Viewpoint'],label='False negatives (FN)')
    ax.barh(y,fp,height=.58,left=fn,color=ACCENT,label='False positives (FP)')
    ax.set_yticks(y,labels=[r['label'] for r in rows]);ax.invert_yaxis()
    ax.set_xlim(0,10.2);ax.set_xlabel('Unmatched event audit entries')
    ax.set_xticks([0,2,4,6,8,10]);ax.grid(axis='x',color='#e1e6e9',lw=.6)
    ax.set_axisbelow(True);ax.tick_params(axis='y',length=0)
    for yy,nf,pf in zip(y,fn,fp):
        if nf:ax.text(nf/2,yy,str(nf),ha='center',va='center',color='white',fontsize=7.1,weight='bold')
        if pf:ax.text(nf+pf/2,yy,str(pf),ha='center',va='center',color='white',fontsize=7.1,weight='bold')
    ax.legend(loc='lower right',bbox_to_anchor=(1,1.0),frameon=False,ncol=2,fontsize=6.7)
    fig.subplots_adjust(left=.34,right=.96,bottom=.23,top=.81)
    save(fig,out,'Fig05_Manually_reviewed_errors')


def plot_virtual_timeline(data:dict,out:Path)->None:
    rows=data['virtual_move_episodes']
    fig,ax=plt.subplots(figsize=(7.05,2.2))
    for i,r in enumerate(rows):
        pred=float(r['pred_start_sec']);y=i
        ax.hlines(y,0,1.1,color=LIGHT,lw=2,zorder=1)
        gt=r.get('gt_start_sec')
        if gt is not None:
            dt=float(gt)-pred
            ax.axvspan(max(0,dt-.5),min(1.1,dt+.5),ymin=(2.9-y)/3.45,ymax=(3.35-y)/3.45,
                       facecolor='#eeeeee',alpha=.75,zorder=0)
            ax.plot(dt,y,marker='^',ms=8,color=COLORS['Interaction'],zorder=4)
            ax.annotate(f'+{dt:.3f} s',xy=(dt,y),xytext=(0,10),textcoords='offset points',
                        ha='center',fontsize=7,color=COLORS['Interaction'])
        ax.plot(0,y,marker='o',ms=7,color=ACCENT,zorder=4)
        if r.get('pred_release_sec') is not None:
            dt=float(r['pred_release_sec'])-pred
            ax.plot(dt,y,marker='x',ms=8,mew=1.5,color='#a45b6b',zorder=4)
            ax.annotate(f'+{dt:.3f} s',xy=(dt,y),xytext=(0,10),textcoords='offset points',
                        ha='center',fontsize=7,color='#a45b6b')
    ax.set_yticks(range(3),labels=[r['name'].split(': ')[1] for r in rows])
    ax.invert_yaxis();ax.set_xlim(-.055,1.065);ax.set_ylim(2.48,-.49)
    ax.set_xlabel('Elapsed time from premature predicted START (s)')
    ax.tick_params(axis='y',length=0)
    ax.grid(axis='x',color='#e1e6e9',lw=.5);ax.set_axisbelow(True)
    handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=ACCENT,
                    markeredgecolor=ACCENT,label='Predicted START'),
             Line2D([0],[0],marker='x',color='none',markeredgecolor='#a45b6b',
                    label='Predicted RELEASE'),
             Line2D([0],[0],marker='^',color='none',markerfacecolor=COLORS['Interaction'],
                    markeredgecolor=COLORS['Interaction'],label='Annotated START')]
    ax.legend(handles=handles,frameon=False,loc='upper left',bbox_to_anchor=(0,1.28),ncol=3)
    fig.subplots_adjust(left=.25,right=.97,bottom=.26,top=.81)
    save(fig,out,'Fig06_Virtual_chess_event_timing')


def plot_prf(data:dict,out:Path)->None:
    s=data['session2']; names=['Precision','Recall','F1']
    vals=[s['precision']*100,s['recall']*100,s['f1']*100]
    fig,ax=plt.subplots(figsize=(3.3,2.25))
    bars=ax.bar(names,vals,width=.60,color=[COLORS['Viewpoint'],COLORS['Working area'],COLORS['Interaction']])
    ax.set_ylim(0,110);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Event-level score (%)')
    ax.grid(axis='y',color='#e1e6e9',lw=.6);ax.set_axisbelow(True)
    for bar,val in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,val+2,f'{val:.1f}%',ha='center',fontsize=7.7)
    fig.subplots_adjust(left=.20,right=.97,bottom=.20,top=.94)
    save(fig,out,'Fig07_Pooled_positive_metrics')


def write_csv(path:Path,headings:list[str],rows:list[list[Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(headings);w.writerows(rows)


def tex_table(path:Path,caption:str,label:str,fmt:str,headers:list[str],rows:list[list[str]])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    lines=[r'\begin{table}[t]',r'\centering',f'\\caption{{{caption}}}',f'\\label{{{label}}}',
           r'\footnotesize',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{'+fmt+'}',r'\hline',
           ' & '.join(headers)+r' \\',r'\hline']
    for r in rows:lines.append(' & '.join(r)+r' \\')
    lines += [r'\hline',r'\end{tabular}',r'\end{table}']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def generate_tables(data:dict,out:Path)->None:
    out.mkdir(parents=True,exist_ok=True)
    abl=sorted(data['ablation'],key=lambda r:r['frames'])
    ab_rows=[[str(r['frames']),f"{r['positive_f1']:.3f}",f"{100*r['positive_recall']:.1f}\\%",
              f"{r['false_start_per_min']:.2f}",f"{r['median_abs_delay_ms']:.1f}"] for r in abl]
    tex_table(out/'Tab01_Confirmation_ablation.tex',
              'Effect of confirmation length on gesture performance (Session 2).',
              'tab:pinch-ablation','lcccc',
              ['Frames','F1','Recall','False/min','Delay (ms)'],ab_rows)
    write_csv(out/'Tab01_Confirmation_ablation.csv',
              ['frames','positive_f1','positive_recall','false_starts_per_min','median_abs_delay_ms'],
              [[r['frames'],r['positive_f1'],r['positive_recall'],r['false_start_per_min'],r['median_abs_delay_ms']] for r in abl])
    s=data['session2']; ss=data['session1']
    overview=[['Positive conditions / clips',f"{ss['positive_test_conditions']} conditions",f"{s['positive_clips']} clips"],
              ['F1',f"{ss['positive_f1_each']:.3f} in all 6 tests",f"{s['f1']:.3f} pooled"],
              ['False START/min',f"{ss['negative_false_starts_per_min']:.2f} (approx.)",f"{s['negative_false_start_per_min']:.2f}"],
              ['Median absolute delay',f"{ss['median_abs_delay_ms']:.0f} ms (approx.)",f"{s['median_abs_delay_ms']:.1f} ms"]]
    tex_table(out/'Tab00_Benchmark_overview.tex',
              'Two distinct benchmark protocols; conditions differ and are not a controlled before/after comparison.',
              'tab:pinch-overview','lll',['Metric','Session 1','Session 2'],overview)
    write_csv(out/'Tab00_Benchmark_overview.csv',['metric','session1','session2'],overview)
    conditions=data['conditions']
    write_csv(out/'Tab02_Conditions.csv',['family','condition','event_f1','hand_detection_pct'],
              [[r['family'],r['condition'],r['f1'],r['hand_detection_pct']] for r in conditions])
    neg=data['negative_controls']
    write_csv(out/'Tab03_Negative_controls.csv',
              ['action','number_of_clips','false_starts','duration_sec','false_starts_per_min'],
              [[r['action'],r['n_clips'],r['false_starts'],r['duration_sec'],round(r['false_start_per_min'],3)] for r in neg])
    neg_tex_rows=[[r['action'],str(r['false_starts']),f"{r['duration_sec']:.2f}",f"{r['false_start_per_min']:.2f}"] for r in neg]
    neg_tex_rows += [['Total',str(s['negative_false_starts']),f"{s['negative_duration_sec']:.2f}",f"{s['negative_false_start_per_min']:.2f}"]]
    tex_table(out/'Tab03_Negative_controls.tex',
              'Unintended PINCH\\_START events during negative controls (Session 2).',
              'tab:pinch-negatives','lrrr',['Action','STARTs','Time (s)','START/min'],neg_tex_rows)
    write_csv(out/'Tab04_Manually_reviewed_errors.csv',['clip_id','fp','fn','manual_finding'],
              [[r['clip_id'],r['fp'],r['fn'],r['note']] for r in data['manually_inspected']])
    episodes=[]
    for r in data['virtual_move_episodes']:
        delta=None if r['gt_start_sec'] is None else float(r['gt_start_sec'])-float(r['pred_start_sec'])
        episodes.append([r['name'],r['pred_start_sec'],r['pred_release_sec'],r['gt_start_sec'],delta])
    write_csv(out/'Tab05_Virtual_chess_episodes.csv',
              ['episode','pred_start_sec','pred_release_sec','annotated_start_sec','start_early_by_sec'],episodes)


def normalize_column(s:str)->str:
    return ''.join(x for x in str(s).lower() if x.isalnum())


def field(df:pd.DataFrame,candidates:list[str])->str|None:
    bynorm={normalize_column(x):x for x in df.columns}
    for name in candidates:
        if normalize_column(name) in bynorm:return bynorm[normalize_column(name)]
    return None


def number(value:Any)->float|None:
    if value is None or (not isinstance(value,(list,dict)) and pd.isna(value)):
        return None
    try:return float(str(value).replace('%','').strip())
    except (TypeError,ValueError):return None


def fraction(v:Any)->float|None:
    f=number(v)
    if f is None:return None
    return f / 100.0 if f>1.00001 else f


def percentage(v:Any)->float|None:
    f=number(v)
    if f is None:return None
    return f*100.0 if f<=1.00001 else f


def filter_three_frames(df:pd.DataFrame)->pd.DataFrame:
    c=field(df,['confirmation_frames','confirmation_frame','confirmation','frame_confirmation',
                'n_confirm','confirm_frames','n_frames','frames','hold_frames','k'])
    if c is not None:
        values=pd.to_numeric(df[c],errors='coerce').dropna()
        # Avoid treating per-clip frame count as the ablation confirmation parameter.
        if len(values)>0 and set(values.unique()).issubset({1,3,5}):
            three=df[pd.to_numeric(df[c],errors='coerce')==3]
            if not three.empty:return three.copy()
    return df.copy()


def condition_key(raw:Any)->str|None:
    s=normalize_column(raw)
    if ('palm' in s and ('orient' in s or 'view' in s or 'facing' in s)) or s in ('palm','palmfacing'):return 'view_palm'
    if ('back' in s and ('view' in s or 'orientation' in s or 'facing' in s) and 'yaw' not in s) or s in ('backofhand','back'):return 'view_back'
    if ('yaw' in s or 'deg' in s) and '75' in s:return 'yaw_75'
    if ('yaw' in s or 'deg' in s) and '45' in s:return 'yaw_45'
    if ('board' in s or 'zone' in s or s in ('left','right','near','far')):
        for token in ('left','right','near','far'):
            if token in s:return 'zone_'+token
    if any(z in s for z in ('virtual','chess','piecemove')):return 'interaction_chess'
    if 'drag' in s:return 'interaction_drag'
    if 'hold' in s or 'stationary' in s:return 'interaction_hold'
    return None


def negative_key(raw:Any)->str|None:
    s=normalize_column(raw)
    # IMPORTANT: use the clip ID, not the misclassified original manifest label.
    if 'fist' in s and 'static' in s:return 'fist_static'
    if 'fist' in s and 'fast' in s:return 'fist_fast'
    if 'fist' in s and 'slow' in s:return 'fist_slow'
    if 'reach' in s or 'hover' in s:return 'reach_hover'
    return None


def integrate_summary_json(data:dict, path:Path, messages:list[str])->None:
    """Safely read pooled Session 2 event counts when a summary JSON exposes them.

    Select only unambiguous positive-event aggregate entries. Do not guess the
    baseline if a file holds several equally plausible confirmation settings.
    """
    obj=json.loads(path.read_text(encoding='utf-8'))
    candidates=[]
    def walk(node:Any, ancestry:str='', depth:int=0)->None:
        if depth>6 or not isinstance(node,dict):return
        keys={normalize_column(k):k for k in node}
        names=[['tp','truepositives','eventtp'],['fp','falsepositives','eventfp'],
               ['fn','falsenegatives','eventfn']]
        selected=[]
        for group in names:
            key=next((keys[k] for k in group if k in keys),None)
            selected.append(number(node[key]) if key is not None else None)
        if all(v is not None and v>=0 and float(v).is_integer() for v in selected):
            tp,fp,fn=(int(x) for x in selected)
            if tp+fn>=50:
                candidates.append((ancestry,node,tp,fp,fn))
        for k,v in node.items():
            if isinstance(v,dict):walk(v,(ancestry+'/'+str(k)).strip('/'),depth+1)
    walk(obj)
    if not candidates:
        messages.append('No pooled positive TP/FP/FN structure recognised in summary.json; retaining handover totals.')
        return
    favored=[x for x in candidates if any(token in x[0].lower().split('/') for token in ['3','3frames','baseline','threeframe'])]
    if favored:candidates=favored
    unique={(x[2],x[3],x[4]) for x in candidates}
    if len(unique)!=1:
        messages.append(f'Ambiguous positive aggregate in summary.json ({len(unique)} possible TP/FP/FN triples); retaining handover totals.')
        return
    _,node,tp,fp,fn=candidates[0]
    prec=tp/(tp+fp) if tp+fp else 0.0
    rec=tp/(tp+fn) if tp+fn else 0.0
    f1=2*prec*rec/(prec+rec) if prec+rec else 0.0
    s=data['session2']
    s.update(tp=tp,fp=fp,fn=fn,annotated_events=tp+fn,
             precision=prec,recall=rec,f1=f1)
    delaycol=next((k for k in node if normalize_column(k) in
                  {'medianabsdelayms','medianabsolutedelayms','medianeventdelayms'}),None)
    if delaycol is not None and number(node[delaycol]) is not None:
        s['median_abs_delay_ms']=number(node[delaycol])
    messages.append(f'UPDATED pooled positive metrics from summary.json: TP={tp}, FP={fp}, FN={fn}, F1={f1:.3f}.')


def integrate_csvs(data:dict,source:Path, strict:bool=False)->list[str]:
    """Safely override values with recognised CSVs, logging any missing fields.

    We never silently reclassify negative_fist_static_01 using a manifest field.
    We cannot guarantee unknown evaluator schemas without the source CSVs, so
    unrecognised files retain the dated handover results with an explicit note.
    """
    messages=[]
    required={'per_condition_metrics.csv','confirmation_ablation.csv','per_clip_metrics.csv'}
    found=set()
    for filename in sorted(required|{'event_matching_audit.csv','summary.json','baseline_replay_consistency.csv'}):
        candidates=list(source.rglob(filename)) if source.exists() else []
        if candidates:
            found.add(filename);messages.append(f'FOUND {filename}: {candidates[0]}')
        else:messages.append(f'ABSENT {filename}')
    summary_paths=list(source.rglob('summary.json'))
    if summary_paths:
        try:integrate_summary_json(data,summary_paths[0],messages)
        except (ValueError,TypeError,KeyError,json.JSONDecodeError) as exc:
            messages.append(f'Could not safely read summary.json ({exc}); retained handover pooled totals.')

    def read_file(fn):
        paths=list(source.rglob(fn));return pd.read_csv(paths[0]) if paths else None

    df=read_file('per_condition_metrics.csv')
    if df is not None:
        messages.append(f'per_condition_metrics.csv columns = {list(df.columns)}')
        df=filter_three_frames(df)
        idcol=field(df,['condition','condition_id','condition_name','test_condition','scenario',
                        'trial_condition','clip_id','clip_name','name'])
        fcol=field(df,['event_f1','event_f1_score','positive_event_f1','positive_f1','f1','f1_score'])
        hcol=field(df,['hand_detection_rate','hand_detection_pct','hand_detect_pct',
                       'hand_presence_pct','hand_detection_rate_pct','hand_detect_rate',
                       'hand_presence_rate','hand_detected_pct','hand_detection'])
        by_key={r['key']:r for r in data['conditions']}
        replaced=set()
        if idcol and fcol:
            for _,r in df.iterrows():
                key=condition_key(r[idcol]); f=fraction(r[fcol])
                if key in by_key and f is not None:
                    by_key[key]['f1']=round(f,5)
                    if hcol and percentage(r[hcol]) is not None:
                        by_key[key]['hand_detection_pct']=round(percentage(r[hcol]),3)
                    replaced.add(key)
            messages.append(f'UPDATED condition F1 from CSV: {len(replaced)}/11 conditions: {sorted(replaced)}')
        else:messages.append('COULD NOT READ condition F1: required condition and F1 columns not recognised')
        if strict and len(replaced)!=11:raise ValueError('Strict CSV mode: could not map all 11 conditions. See column names in log.')
        # If each condition has explicit event counts, reconcile against the pooled
        # session result. This only recomputes from non-overlapping, complete rows.
        tpcol=field(df,['tp','true_positives','event_tp','positive_tp'])
        fpcol=field(df,['fp','false_positives','event_fp','positive_fp'])
        fncol=field(df,['fn','false_negatives','event_fn','positive_fn'])
        if len(replaced)==11 and idcol and tpcol and fpcol and fncol:
            counts={}
            for _,r in df.iterrows():
                key=condition_key(r[idcol])
                nums=[number(r[col]) for col in (tpcol,fpcol,fncol)]
                if key in replaced and all(x is not None for x in nums):
                    counts.setdefault(key,[]).append(nums)
            if len(counts)==11 and all(len(v)==1 for v in counts.values()):
                tp=int(sum(v[0][0] for v in counts.values()))
                fp=int(sum(v[0][1] for v in counts.values()))
                fn=int(sum(v[0][2] for v in counts.values()))
                if (tp,fp,fn)==(data['session2']['tp'],data['session2']['fp'],data['session2']['fn']):
                    messages.append(f'VERIFIED pooled TP/FP/FN against condition rows: {tp}/{fp}/{fn}.')
                else:
                    messages.append(f'CHECK pooled condition counts {tp}/{fp}/{fn} differ from dated handover totals. No pooled override without confirmation of non-overlapping rows.')

    df=read_file('confirmation_ablation.csv')
    if df is not None:
        messages.append(f'confirmation_ablation.csv columns = {list(df.columns)}')
        ncol=field(df,['confirmation_frames','confirmation','confirmation_frame','frames',
                       'n_confirm','confirm_frames','n_frames','hold_frames','k'])
        fcol=field(df,['positive_f1','positive_event_f1','event_f1','f1','f1_score'])
        rcol=field(df,['positive_recall','event_recall','recall'])
        negcol=field(df,['false_start_per_min','false_starts_per_min',
                         'negative_false_starts_per_min','negative_fp_start_per_min',
                         'false_pinch_start_per_min'])
        delaycol=field(df,['median_abs_delay_ms','median_absolute_delay_ms',
                           'median_event_delay_ms','median_delay_ms','event_delay_ms'])
        by_n={int(r['frames']):r for r in data['ablation']}
        replaced=set()
        if ncol and fcol and rcol and negcol and delaycol:
            for _,r in df.iterrows():
                n=number(r[ncol]);n=int(n) if n is not None else -1
                if n in by_n:
                    f=fraction(r[fcol]);rec=fraction(r[rcol]); neg=number(r[negcol]);delay=number(r[delaycol])
                    if all(v is not None for v in (f,rec,neg,delay)):
                        by_n[n].update(positive_f1=f,positive_recall=rec,false_start_per_min=neg,median_abs_delay_ms=delay)
                        replaced.add(n)
            messages.append(f'UPDATED ablation from CSV for frames: {sorted(replaced)}')
        else:messages.append('COULD NOT READ ablation: expected frames, positive F1/recall, negative START/min, delay (ms).')
        if strict and replaced!={1,3,5}:raise ValueError('Strict CSV mode: ablation CSV not fully parsed.')

    df=read_file('per_clip_metrics.csv')
    if df is not None:
        messages.append(f'per_clip_metrics.csv columns = {list(df.columns)}')
        df=filter_three_frames(df)
        ccol=field(df,['clip_id','clip_name','video_id','clip','video','filename','file'])
        countcol=field(df,['false_start_count','false_starts','false_pinch_starts','false_pinch_start_count',
                           'negative_false_starts','unintended_starts','unmatched_start_count','n_false_starts'])
        durcol=field(df,['duration_s','duration_sec','duration_seconds','clip_duration_s',
                         'recording_duration_s','video_duration_s','duration'])
        groups:dict[str,list[tuple[float,float]]]=defaultdict(list)
        if ccol and countcol and durcol:
            for _,r in df.iterrows():
                key=negative_key(r[ccol]);count=number(r[countcol]);dur=number(r[durcol])
                if key and count is not None and dur is not None and dur>0:groups[key].append((count,dur))
            if set(groups)==set(r['key'] for r in data['negative_controls']):
                for r in data['negative_controls']:
                    vals=groups[r['key']];r['false_starts']=int(round(sum(x[0] for x in vals)))
                    r['duration_sec']=round(sum(x[1] for x in vals),4)
                    r['n_clips']=len(vals)
                s=data['session2'];s['negative_false_starts']=sum(r['false_starts'] for r in data['negative_controls'])
                s['negative_duration_sec']=sum(r['duration_sec'] for r in data['negative_controls'])
                messages.append('UPDATED all negative-control rates from per_clip CSV; static-fist clip identified by filename.')
            else:messages.append(f'INCOMPLETE negative-control groups ({list(groups)}); retained handover negative results.')
        else:messages.append('COULD NOT READ negative counts/durations from per_clip CSV; retained handover negative results.')
        if strict and set(groups)!=set(r['key'] for r in data['negative_controls']):
            raise ValueError('Strict CSV mode: cannot parse all four negative groups.')
    return messages


def extra_csv_plots(source:Path,out:Path,messages:list[str])->None:
    """Generate optional diagnostics only when real numerical source data exist."""
    audit=list(source.rglob('event_matching_audit.csv'))
    if audit:
        df=pd.read_csv(audit[0]);messages.append('Audit columns: '+str(list(df.columns)))
        status=field(df,['match_status','status','outcome','result','match_result'])
        delay_ms=field(df,['delay_ms','signed_delay_ms','event_delay_ms','timing_error_ms'])
        delay_s=field(df,['delay_s','signed_delay_s','event_delay_s','timing_error_s','matched_delay_s'])
        pred=field(df,['pred_time_s','prediction_time_s','detected_time_s','predicted_time_s'])
        gt=field(df,['annotation_time_s','gt_time_s','ground_truth_time_s','label_time_s'])
        if delay_ms:delays=pd.to_numeric(df[delay_ms],errors='coerce')
        elif delay_s:delays=1000*pd.to_numeric(df[delay_s],errors='coerce')
        elif pred and gt:delays=1000*(pd.to_numeric(df[pred],errors='coerce')-pd.to_numeric(df[gt],errors='coerce'))
        else:delays=None
        if delays is not None:
            if status:
                mask=df[status].astype(str).str.lower().str.contains('match|true|^tp$',regex=True)
                mask &= ~df[status].astype(str).str.lower().str.contains('unmatch')
                delays=delays[mask]
            delays=delays[np.isfinite(delays)]
            if len(delays)>=10:
                fig,ax=plt.subplots(figsize=(3.48,2.45))
                ax.hist(delays,bins=35,color=COLORS['Viewpoint'],alpha=.85,edgecolor='white',lw=.3)
                ax.axvline(0,color=DARK,lw=.85,ls='--')
                ax.set_xlabel('Signed event delay (ms)')
                ax.set_ylabel('Matched events')
                ax.grid(axis='y',color='#e1e6e9',lw=.5);ax.set_axisbelow(True)
                fig.subplots_adjust(left=.17,right=.97,top=.95,bottom=.22)
                save(fig,out,'Fig08_Matched_event_delay_distribution')
                messages.append(f'CREATED delay histogram from {len(delays)} matched event delays.')
            else:messages.append('SKIPPED delay histogram: <10 matched delay measurements available.')
        else:messages.append('SKIPPED delay histogram: audit delay columns unrecognised.')

    clip=list(source.rglob('per_clip_metrics.csv'))
    if clip:
        df=filter_three_frames(pd.read_csv(clip[0]))
        name=field(df,['clip_id','clip_name','clip','filename','video'])
        med=field(df,['processing_median_ms','median_processing_ms','median_process_ms',
                      'frame_processing_median_ms','median_compute_ms','processing_time_median_ms'])
        p95=field(df,['processing_p95_ms','p95_processing_ms','p95_process_ms',
                      'frame_processing_p95_ms','processing_time_p95_ms','p95_compute_ms'])
        if name and med and p95:
            xy=[(str(r[name]),number(r[med]),number(r[p95])) for _,r in df.iterrows()]
            xy=[r for r in xy if r[1] is not None and r[2] is not None]
            if xy:
                fig,ax=plt.subplots(figsize=(7.0,max(2.2,.25*len(xy)+.5)))
                y=np.arange(len(xy))
                ax.plot([r[1] for r in xy],y,'o',color=COLORS['Viewpoint'],ms=3,label='Median')
                ax.plot([r[2] for r in xy],y,'s',color=ACCENT,ms=3,label='95th percentile')
                for yi,(_,m,p) in enumerate(xy):ax.hlines(yi,m,p,color=LIGHT,lw=2,zorder=1)
                ax.set_yticks(y,labels=[r[0].replace('_01','').replace('_',' ') for r in xy])
                ax.invert_yaxis();ax.set_xlabel('Per-frame computation time (ms)')
                ax.legend(frameon=False,ncol=2,loc='upper right')
                ax.grid(axis='x',color='#e1e6e9',lw=.5);ax.set_axisbelow(True)
                fig.subplots_adjust(left=.38,right=.96,bottom=.12,top=.96)
                save(fig,out,'Fig09_Per_clip_processing_time')
                messages.append(f'CREATED processing-time plot from {len(xy)} clips.')
        else:messages.append('SKIPPED processing-time plot: per-clip median and p95 columns unrecognised.')


def video_contact_sheets(video_root:Path,out:Path,messages:list[str],nframes:int=12)->None:
    try:import cv2
    except ImportError:
        messages.append('SKIPPED video contact sheets: pip install opencv-python');return
    if not video_root.exists():
        messages.append(f'Video folder not found: {video_root}');return
    identifiers=['negative_fist_fast_transition_01', 'orientation_back_yaw_75deg_01',
                 'board_zone_left_01','interaction_virtual_piece_move_01']
    formats={'.mp4','.mov','.avi','.mkv','.m4v'}
    files=[p for p in video_root.rglob('*') if p.is_file() and p.suffix.lower() in formats]
    out.mkdir(parents=True,exist_ok=True)
    for identity in identifiers:
        matches=[p for p in files if identity in p.stem]
        if not matches:
            messages.append(f'Video not found for contact sheet: {identity}');continue
        path=matches[0];cap=cv2.VideoCapture(str(path))
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS))
        if count<4 or fps<=0:
            messages.append(f'Could not read video frames: {path.name}');cap.release();continue
        indices=np.round(np.linspace(int(count*.04),int(count*.96),nframes)).astype(int)
        fig,axs=plt.subplots(math.ceil(nframes/4),4,figsize=(9.1,math.ceil(nframes/4)*1.8))
        axs=np.ravel(axs)
        successes=0
        for ax,idx in zip(axs,indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx));ok,bgr=cap.read()
            if ok:
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                ax.imshow(rgb);successes+=1
            ax.axis('off')
            ax.set_title(f'{idx/fps:.2f} s  |  frame {idx}',fontsize=7)
        cap.release()
        for ax in axs[len(indices):]:ax.axis('off')
        fig.suptitle(f'{identity}  |  inspect frames against event audit',fontsize=9)
        fig.tight_layout(rect=(0,0,1,.96))
        fig.savefig(out/f'Contact_sheet_{identity}.png',dpi=220,bbox_inches='tight')
        plt.close(fig)
        messages.append(f'CREATED contact sheet from {successes}/{nframes} frames: {path.name}')
    messages.append('Contact sheets are sampling aids only; manually inspect video to identify causal moments.')


def write_readme(data:dict, out:Path, csv_dir:Path|None, messages:list[str])->None:
    s=data['session2'];negative=s['negative_false_start_per_min']
    lines=[
        '# AR chess pinch-gesture benchmark: figure package','',
        'Generated from the consolidated gesture benchmark handover (6 November 2026).',
        'Figures are publication-sized. Use PDF for LaTeX and PNG for preview.',
        '',
        '## Suggested main-report materials','',
        '1. `main/Fig01_F1_by_condition.pdf`: session 2 event-level F1 by condition.',
        '2. `main/Fig02_False_START_rates.pdf`: negative controls, corrected static-fist category.',
        '3. `tables/Tab01_Confirmation_ablation.tex`: small 1/3/5-frame table.',
        '',
        'If space is particularly tight, combine Figs 01 and 02 or use only Fig 01 + the ablation table.',
        '',
        '## Supplemental diagnostics','',
        '- `supplementary/Fig03_Confirmation_ablation.pdf`: graph version of the ablation table.',
        '- `supplementary/Fig04_Hand_tracking_vs_F1.pdf`: high hand presence can coexist with low event recognition.',
        '- `supplementary/Fig05_Manually_reviewed_errors.pdf`: three manually reviewed clips, not all failure cases.',
        '- `supplementary/Fig06_Virtual_chess_event_timing.pdf`: three root episodes for six audit entries.',
        '- `supplementary/Fig07_Pooled_positive_metrics.pdf`: precision/recall/F1 summary.',
        '- Figs 08 and 09 are created only when the relevant raw evaluator CSVs are readable.',
        '- `tables/`: editable CSVs and ready-to-include LaTeX tables.',
        '- `video_contact_sheets/`: created only when `--videos` points to the real videos.',
        '',
        '## Tested findings (3-frame baseline)','',
        f"- Session 1 (six controlled positive conditions): F1 1.000 in each; negative {data['session1']['negative_false_starts_per_min']:.3f} START/min (approximate from earlier plot).",
        f"- Session 2: {s['tp']} TP, {s['fp']} FP, {s['fn']} FN across {s['annotated_events']} annotations; precision {s['precision']:.3f}, recall {s['recall']:.3f}, F1 {s['f1']:.3f}.",
        f"- Session 2 negative controls: {s['negative_false_starts']} false STARTs over {s['negative_duration_sec']:.2f} s = {negative:.2f}/min.",
        f"- Session 2 median absolute matched-event delay: {s['median_abs_delay_ms']:.1f} ms; based on capture timestamps, not full end-to-end display latency.",
        '',
        '## Critical interpretation safeguards','',
        '- Sessions 1 and 2 use different difficulty levels and negative-action compositions: do **not** treat them as a controlled before/after comparison.',
        '- The positive F1 is an **event** metric, not the percentage of full chess moves completed successfully.',
        '- The negative START rate is per minute, not per fist closure; faster repetitions increase the number of opportunities per minute.',
        '- A genuine physical fingertip contact during a fist action can still be an unintended chess command; intention and contact are different labels.',
        '- The hand-detection percentages rely on hand-visibility annotations; review boundary visibility before claiming precise tracking error rates.',
        '- Correct `negative_fist_static_01` to `fist_static` in `session02/manifest.csv` before using evaluator-generated category charts.',
        '- Virtual chess movement has 4 FP + 2 FN audit entries, traced to three premature START root episodes, not six independent causes.',
        '- No reliable ground-truth frame image can be generated from the numerical handover alone. Use video contact sheets for manual selection.',
        '- All timestamps and condition diagnoses here are based on the supplied handover. Verify against source data before final submission.',
        '',
        '## Re-run','',
        '```bash',
        'pip install numpy pandas matplotlib opencv-python',
        'python generate_pinch_benchmark_results.py --out pinch_results',
        'python generate_pinch_benchmark_results.py --session2-dir path/to/session02 --out pinch_results',
        'python generate_pinch_benchmark_results.py --videos path/to/videos --out pinch_results',
        'python generate_pinch_benchmark_results.py --data-json pinch_benchmark_data.json --out pinch_results',
        '```','',
        'For custom data, edit a copy of `pinch_benchmark_data.json` and provide it using `--data-json`.',
        'The script uses the handover values unless a recognisable local CSV or unambiguous summary.json explicitly updates a metric.',
        'CSV parsing depends on your evaluator column names. Consult `data_provenance.txt` to see exactly what was read.',
        'Use `--strict-csv` to reject incomplete primary CSV parsing instead of retaining handover values.',
        '',
        '## Source and parsing log',''
    ]
    lines.extend('- '+message for message in messages)
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out',type=Path,default=Path('pinch_benchmark_results'),help='Output directory')
    parser.add_argument('--data-json',type=Path,help='Override the packaged numerical handover values with a JSON file')
    parser.add_argument('--session2-dir',type=Path,help='Optional evaluator directory containing Session 2 CSVs')
    parser.add_argument('--strict-csv',action='store_true',help='Fail if the main CSVs cannot be completely parsed')
    parser.add_argument('--videos',type=Path,help='Optional video directory, to generate inspection contact sheets')
    parser.add_argument('--contact-sheet-frames',type=int,default=12,help='Frames per video contact sheet')
    args=parser.parse_args()
    data=copy.deepcopy(SOURCE_DATA)
    if args.data_json:
        data=json.loads(args.data_json.read_text(encoding='utf-8'))
    messages=[f"Base numerical source: {data.get('source','external user JSON')}"]
    if args.session2_dir:
        messages.extend(integrate_csvs(data,args.session2_dir,strict=args.strict_csv))
    elif args.strict_csv:
        parser.error('--strict-csv requires --session2-dir')
    calculate(data);validate(data);setup_style()
    root=args.out;main_dir=root/'main';supp=root/'supplementary';tables=root/'tables'
    plot_condition_f1(data,main_dir)
    plot_negative(data,main_dir)
    plot_ablation(data,supp)
    plot_tracking_vs_f1(data,supp)
    plot_audited_errors(data,supp)
    plot_virtual_timeline(data,supp)
    plot_prf(data,supp)
    generate_tables(data,tables)
    if args.session2_dir:extra_csv_plots(args.session2_dir,supp,messages)
    if args.videos:video_contact_sheets(args.videos,root/'video_contact_sheets',messages,max(4,args.contact_sheet_frames))
    (root/'resolved_plot_data.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    (root/'data_provenance.txt').write_text('\n'.join(messages)+'\n',encoding='utf-8')
    write_readme(data,root,args.session2_dir,messages)
    print(f'Created figures and LaTeX tables under {root.resolve()}')
    for msg in messages:print('  '+msg)


if __name__=='__main__':main()
