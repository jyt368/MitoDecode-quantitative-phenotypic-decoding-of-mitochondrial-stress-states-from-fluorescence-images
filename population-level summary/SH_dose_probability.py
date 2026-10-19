import torch
import torch.nn as nn
import argparse
from transformer_4_4classes import ResNet50_ViT
import torchvision
from torch.utils.data import DataLoader
#need to change to data_handler_224_mito_inf when testing infinitus compounds; else, use data_handler_224_mito_plot
from data_handler import Get_Dataset, CustomTransform, ToTensor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
import torch.nn.functional as F
from scipy.interpolate import PchipInterpolator
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f_oneway


def load_opt():
    opt= argparse.Namespace()
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.train_path=None
    opt.test_batch_size=1
    return opt


def load_model(model_path):
    recall= torch.load(model_path)
    model= ResNet50_ViT()
    model.load_state_dict(recall)
    model.to(opt.device)
    model.eval()
    return model

def get_probabilities(model, group_paths, group_name):  

    total = []
    data=[]
    probabilities = {f'Group{i+1}': [] for i in range(len(group_name))} 
    max_score=0

    with torch.no_grad():
        for i, group in enumerate(group_paths):
            dataset=Get_Dataset(group,Transform=ToTensor())
            test_loader = DataLoader(dataset=dataset,
                                    batch_size=opt.test_batch_size,
                                    shuffle=False,
                                    num_workers=0)
            total.append(len(dataset))
            print(f'Group {i+1} has {len(dataset)} samples.')
            for inputs,labels in test_loader:
                inputs,_=inputs.to(opt.device),labels.to(opt.device)

                output=model(inputs)
                _, preds = torch.max(output, 1)
                probs = F.softmax(output, dim=1)
                print(probs.shape)

                #score= probs[0,1].item()+probs[0,2].item()*2+probs[0,3].item()*3
                t=int(preds)
                if t==0:
                    score= probs[0,3].item()
                else:
                    score= probs[0,3].item()
                
                #score= probs[0,1].item()+probs[0,2].item()*2+probs[0,3].item()*3

                max_score = max(max_score, score)

                '''    score=0
                    for j in range(1,t+1):
                        score += probs[0,j].item()*j'''
                print('score:', (t, score))

                probabilities[f'Group{i+1}'].append(score)
                data.append({'group': group_name[i], 'value': score})

    # normalise if needed
    '''for group in probabilities:
            probabilities[group] = [score / max_score if max_score > 0 else 0 for score in probabilities[group]]
    for entry in data:
            entry['value'] = entry['value'] / max_score if max_score > 0 else 0'''

    pack=pd.DataFrame(data)
    # print(pack.head())
    pack['value'] = pack['value'] * 100 # Convert to percentage

    for group in group_name:
        group_data = pack[pack['group'] == group]['value']
        print(f"Statistics for {group}:")
        print(f"  Mean: {group_data.mean():.4f}")
        print(f"  Median: {group_data.median():.4f}")
        print(f"  Std Dev: {group_data.std():.4f}")
        print(f"  Min: {group_data.min():.4f}")
        print(f"  Max: {group_data.max():.4f}\n")
    return pack

def run_oneway_anova(pack, group_name):
    groups = [pack[pack['group']==g]['value'] for g in group_name]
    F, p = f_oneway(*groups)
    print(f"\nOne-way ANOVA: F={F:.4f}, p={p:.4e}")
    return F, p

def run_compare(pack, group_name, control='Control'):
    # one-way ANOVA
    groups = [
        pack.loc[pack['group'] == g, 'value'].dropna()
        for g in group_name
    ]

    F, p_anova = f_oneway(*groups)
    print(f"One-way ANOVA: F={F:.3f}, p={p_anova:.4g}")

    # Tukey post hoc
    tukey = pairwise_tukeyhsd(
        endog=pack['value'],
        groups=pack['group'],
        alpha=0.05
    )

    # only comparisons with the control
    results = {}
    pairs = []

    for row in tukey.summary().data[1:]:
        g1, g2, meandiff, p_adj, lower, upper, reject = row

        if control in (g1, g2):
            pair = (g1, g2)
            pairs.append(pair)
            results[pair] = float(p_adj)

    return p_anova, pairs, results

def p_to_star(p):
    if  p < 0.0001:
        return '****'
    elif p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'

def prob_plot(pack, group_name):
    sns.set(style="white", context="talk", font_scale=1.3)
    plt.figure(figsize=(8,8))
    blue_palette= sns.color_palette("blend:#9DC5BC,#6FAF9F", n_colors=len(group_name))
    '''
    grey: "blend:#C0BEBE,#B8B8B8"
    blue: "blend:#D0DFF5,#6F8FBF"
    purple: "blend:#B9A1B9,#8C79B8"
    green: "blend:#9DC5BC,#6FAF9F"

    '''
    print("Plotting...")
    order = group_name

    ax = sns.boxplot(
        data=pack,
        x="group",
        y="value",
        order=order,
        palette=blue_palette,
        boxprops=dict(edgecolor='black'),
        flierprops=dict(marker=''),
        zorder=1
    )
    sns.stripplot(
        data=pack,
        x="group",
        y="value",
        order=order,
        color="black",
        size=2.5,
        jitter=0.25,
        alpha=0.8,
        ax=ax,
        zorder=2
    )

    means = pack.groupby('group')['value'].mean()
    sems = pack.groupby('group')['value'].sem()
    group_order = pack['group'].unique()
    x_positions = np.arange(len(group_order))

    ax.scatter(x_positions, means[group_order], color="#4A4A4A" , marker='D', s=50, zorder=4, label='Mean')
    '''ax.errorbar(
        x_positions,
        means[group_order], 
        yerr=sems[group_order],
        fmt='o',
        color="#4A4A4A" ,
        markersize=7,
        capsize=4,
        linewidth=1.8,
        label="Mean ± SEM",
        zorder=5
    )'''
    
    anova_p, pairs, results = run_compare(pack,group_name,control='Control')
    print(f"Overall ANOVA p = {anova_p:.4g}")

    '''for i, (g1, g2) in enumerate(pairs):
        p = results[(g1,g2)]
        #star = p_to_star(p)

        x1 = group_name.index(g1)
        x2 = group_name.index(g2)
        y = pack['value'].max() + (i+1)*5

        ax.plot([x1, x1, x2, x2], [y, y+2, y+2, y], lw=1.5, c='black')
        #ax.text((x1+x2)/2, y+1, f"{star}", ha='center', va='bottom', fontsize=18)
    '''    
    y_max = pack['value'].max()
    n_comparisons = len(pairs)
    ax.spines['left'].set_visible(True)
    ax.yaxis.set_ticks_position('left')
    ax.set_ylim(-1, y_max + 5)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.tick_params(axis='y', which='major', length=6, width=1.2, direction='out')

    plt.xticks(rotation=30, ha='right', color='black')
    sns.despine()
    plt.ylabel("Rotenone-like Phenotypic Probability (%)")
    plt.tight_layout()
    plt.show()


'''------------------- Main ------------------'''
opt = load_opt()
pth= 'Z:/Users/yj368/ROI isentification/Health_state_prediction/organised training code/work for mito and lyso identification task/Code_for_publication/trained_models/SH_4classes.pt'
model=load_model(pth)

'''group_name = ['Control', 'H2O2 10uM', 'H2O2 50uM', 'H2O2 100uM', 'H2O2 500uM', 'H2O2 1mM']
group_paths = [
    'Z:/Users/yj368/Mitochondria classification data/SHSY5Y/Control/Indiv2/Control_test_all.txt',
    'Z:/Users/yj368/ROI isentification/SH/H2O2/correct/Indiv/10uM.txt',
    'Z:/Users/yj368/ROI isentification/SH/H2O2/correct/Indiv/50uM.txt',
    'Z:/Users/yj368/ROI isentification/SH/H2O2/correct/Indiv/100uM.txt',
    'Z:/Users/yj368/ROI isentification/SH/H2O2/correct/Indiv/500uM.txt',
    'Z:/Users/yj368/ROI isentification/SH/H2O2/correct/Indiv/1mM_test.txt'
]'''

'''group_name = ['Control', 'FCCP 0.1uM', 'FCCP 0.5uM', 'FCCP 1uM', 'FCCP 5uM']
group_paths = [
    'Z:/Users/yj368/Mitochondria classification data/SHSY5Y/Control/Indiv2/Control_test_all.txt',
    'Z:/Users/yj368/ROI isentification/SH/FCCP/Indiv/0.1uM.txt',
    'Z:/Users/yj368/ROI isentification/SH/FCCP/Indiv/0.5uM.txt',
    'Z:/Users/yj368/ROI isentification/SH/FCCP/Indiv/1uM.txt',
    'Z:/Users/yj368/ROI isentification/SH/FCCP/Indiv/5uM_test.txt',
]'''

group_name = ['Control', 'Rotenone 0.1uM', 'Rotenone 5uM', 'Rotenone 10uM']
group_paths = [
    'Z:/Users/yj368/Mitochondria classification data/SHSY5Y/Control/Indiv2/Control_test_all.txt',
    'Z:/Users/yj368/ROI isentification/SH/Rotenone/Rotenone/Actin/Indiv/0.1uM.txt',
    'Z:/Users/yj368/ROI isentification/SH/Rotenone/Rotenone/Actin/Indiv/5uM.txt',
    'Z:/Users/yj368/ROI isentification/SH/Rotenone/Rotenone/Actin/Indiv/10uM_test.txt'
]


'''group_name = ['Control', 'Tunicamycin 0.1ug', 'Tunicamycin 1ug', 'Tunicamycin 5ug', 'Tunicamycin 10ug']
group_paths = [
    'Z:/Users/yj368/ROI isentification/SH/Control/Indiv/Control.txt',
    'Z:/Users/yj368/ROI isentification/SH/Tunicamycin/Tunicamycin/Actin/Indiv/0.1ug.txt',
    'Z:/Users/yj368/ROI isentification/SH/Tunicamycin/Tunicamycin/Actin/Indiv/1ug.txt',
    'Z:/Users/yj368/ROI isentification/SH/Tunicamycin/Tunicamycin/Actin/Indiv/5ug.txt',
    'Z:/Users/yj368/ROI isentification/SH/Tunicamycin/Tunicamycin/Actin/Indiv/10ug_test.txt'
]'''

'''group_name = ['Control', 'NAC only', '500uM H2O2', '2.5mM NAC + 500uM H2O2']
group_paths = [
    'Z:/Users/yj368/ROI isentification/SH/NAC/24012026_NAC2.5uM/Indiv/control.txt',
    'Z:/Users/yj368/ROI isentification/SH/NAC/24012026_NAC2.5uM/Indiv/NAC.txt',
    'Z:/Users/yj368/ROI isentification/SH/H2O2/correct/Indiv/500uM.txt',
    'Z:/Users/yj368/ROI isentification/SH/NAC/NAC_H2O2_all.txt'
]'''

'''group_name = ['Control','2.5uM PFFs 24 hours', '2.5uM PFFs 48hours']
group_paths = [
    'Z:/Users/yj368/ROI isentification/SH/Control/Indiv/Control.txt',
    'Z:/Users/yj368/ROI isentification/SH/PFFs/24hour_all.txt',
    'Z:/Users/yj368/ROI isentification/SH/PFFs/48hour_all.txt'
]'''
'''group_name = ['Control', '2.5uM PFFs 48hrs repeat1', '2.5uM PFFs 48hrs repeat2', '2.5uM PFFs 48hrs repeat3']
group_paths = [
    'Z:/Users/yj368/ROI isentification/SH/Control/Indiv/Control.txt',
    'Z:/Users/yj368/ROI isentification/SH/PFFs/48hours/Indiv/repeat1.txt',
    'Z:/Users/yj368/ROI isentification/SH/PFFs/48hours/Indiv/repeat2.txt',
    'Z:/Users/yj368/ROI isentification/SH/PFFs/48hours/Indiv/repeat3.txt'
]'''

'''group_name = ['Control', '2.5uM PFFs 24hrs repeat1', '2.5uM PFFs 24hrs repeat2', '2.5uM PFFs 48hrs']
group_paths = [
    'Z:/Users/yj368/ROI isentification/SH/Control/Indiv/Control.txt',
    'Z:/Users/yj368/ROI isentification/SH/PFFs/24hours/Indiv/2.5uM_repeat1.txt',
    'Z:/Users/yj368/ROI isentification/SH/PFFs/24hours/Indiv/2.5uM_repeat2.txt',
    'Z:/Users/yj368/ROI isentification/SH/PFFs/48hours/Indiv/repeat1.txt'
]'''
n_classes = 4
class_types = ['Control', 'FCCP', 'H2O2', 'Rotenone']

class_colors = {
    'Control':   '#B8B8B8',  # light gray
    'FCCP':      '#6F8FBF',  # soft blue
    'H2O2':      '#8C79B8',  # soft purple
    'Rotenone':  '#6FAF9F',  # soft green
}
pack=get_probabilities(model, group_paths, group_name)
run_oneway_anova(pack, group_name)
# pairs, results = run_compare(pack, group_name, control='Control')
prob_plot(pack, group_name)
