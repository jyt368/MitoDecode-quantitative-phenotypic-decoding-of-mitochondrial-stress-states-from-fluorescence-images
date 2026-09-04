import torch
import torch.nn as nn
import argparse
from transformer_4_6classes import ResNet50_ViT
import torchvision
from torch.utils.data import DataLoader
#need to change to data_handler_224_mito_inf when testing infinitus compounds; else, use data_handler_224_mito_plot
from data_handler_224_SH_6classes import Get_Dataset, CustomTransform, ToTensor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

def load_opt():
    opt= argparse.Namespace()
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.train_path=None
    opt.test_batch_size=1
    return opt


def load_model(model_path):
    recall= torch.load(model_path)
    model= ResNet50_ViT()
    model.load_state_dict(recall['model'])
    print(recall['best_acc'])
    model.to(opt.device)
    model.eval()
    return model



def get_probabilities(model, group_paths, group_name):
    all_probs = []  

    with torch.no_grad():
        for i, group in enumerate(group_paths):
            dataset = Get_Dataset(group, Transform=ToTensor())
            loader = DataLoader(dataset, batch_size=1, shuffle=False)

            for inputs, labels in loader:
                inputs = inputs.to(opt.device)

                output = model(inputs)
                probs = F.softmax(output, dim=1).cpu().numpy()[0]  
                print(f'Group: {group_name[i]}, Probabilities: {probs}')

                for c in range(n_classes):
                    all_probs.append({
                        'group': group_name[i],
                        'class': class_types[c],
                        'probability': probs[c]
                    })
    return all_probs

def cal_mean(all_probs, group_name, class_types):
    df = pd.DataFrame(all_probs)
    df['group'] = pd.Categorical(
        df['group'],
        categories=group_name,
        ordered=True
    )
    print(df.head())

    # mean probability
    mean_df = (
        df
        .groupby(['group', 'class'])['probability']
        .mean()
        .reset_index()
    )

    pivot_df = mean_df.pivot(index='group', columns='class', values='probability')
    pivot_df = pivot_df.loc[group_name, class_types]
    pivot_df = pivot_df * 100 # convert to percentage

    return pivot_df

def plot_stacked_bar(pivot_df, class_types, class_colors):
    sns.set(style="white", context="talk", font_scale=1.3)
    plt.figure(figsize=(12, 7))
    bottom = np.zeros(len(pivot_df))
    x = np.arange(len(pivot_df))
    
    for cls in class_types:
        plt.bar(
            x,
            pivot_df[cls],
            bottom=bottom,
            label=cls,
            color=class_colors[cls],
            edgecolor='black',
            linewidth=0.5
        )
        bottom += pivot_df[cls].values

    ax = plt.gca()
    ax.set_yticks(np.arange(0, 101, 20))
    ax.tick_params(axis='y', which='major', length=6, width=1.5, direction='out', left=True)

    plt.xticks(x, pivot_df.index, rotation=30, ha='right')
    plt.ylabel('Mean Classification Probability (%)')
    # plt.title('Changes in Classification Probability Across Stress Types with the Increase of Dose')
    #plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.legend(
        title='Class',
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        frameon=False
    )
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_wasserstein_distance(pivot_df, group_name):
    P = pivot_df.values
    n = len(group_name)
    w_matrix = np.zeros((n, n))

    x = np.arange(P.shape[1])
    for i in range(n):
        for j in range(n):
            w = wasserstein_distance(
                x, x,
                u_weights=P[i],
                v_weights=P[j]
            )
            w_matrix[i, j] = w

    sns.set(style="white", context="talk", font_scale=1.1)
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        w_matrix,
        xticklabels=group_name,
        yticklabels=group_name,
        cmap=sns.light_palette('#F0A0C0', as_cmap=True),
        square=True,
        cbar_kws={'label': 'Wasserstein distance'},
        linewidths=0.5
    )

    plt.title('Earth Mover’s Distance Between Probability Distributions')
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)

    sns.despine()
    plt.tight_layout()
    plt.show()

'''---main---'''
opt = load_opt()
pth= 'Z:/Users/yj368/ROI isentification/Health_state_prediction/organised training code/work for mito and lyso identification task/ResNet_mito_new_model/Stress Mechanism/SH/Final_PD/transResNet_multitask_6classes_SH_2.pt'
model=load_model(pth)

'''group_name = ['Control', 'FCCP 0.1uM', 'FCCP 0.5uM','FCCP 1uM', 'FCCP 5uM', 'H2O2 1uM', 'H2O2 5uM', 'H2O2 10uM','Heat Shock']
group_paths = [
    'Z:/Users/yj368/ROI isentification/healthy_unhealthy_cells/data/FCCP concentration/0.5uM/train_control_1.txt',
    'Z:/Users/yj368/ROI isentification/healthy_unhealthy_cells/data/FCCP concentration/0.1uM/treated_ML_individual/train_frag.txt',
    'Z:/Users/yj368/ROI isentification/healthy_unhealthy_cells/data/FCCP concentration/0.5uM/train_frag.txt',
    'Z:/Users/yj368/ROI isentification/healthy_unhealthy_cells/data/FCCP concentration/1uM/train_frag_1.txt',
    'Z:/Users/yj368/ROI isentification/healthy_unhealthy_cells/data/FCCP concentration/5uM/train_frag_1.txt'
]'''

'''group_name = ['Control', 'H2O2 10uM', 'H2O2 50uM', 'H2O2 100uM', 'H2O2 500uM', 'H2O2 1mM']
group_paths = [
    'Z:/Users/yj368/ROI isentification/SH/Control/Indiv2_test.txt',
    'Z:/Users/yj368/ROI isentification/SH/H2O2/correct/Indiv/10uM.txt',
    'Z:/Users/yj368/ROI isentification/SH/H2O2/correct/Indiv/50uM.txt',
    'Z:/Users/yj368/ROI isentification/SH/H2O2/correct/Indiv/100uM.txt',
    'Z:/Users/yj368/ROI isentification/SH/H2O2/correct/Indiv/500uM.txt',
    'Z:/Users/yj368/ROI isentification/SH/H2O2/correct/Indiv/1mM_test.txt'
]'''

'''group_name = ['Control', 'FCCP 0.1uM', 'FCCP 0.5uM', 'FCCP 1uM', 'FCCP 5uM']
group_paths = [
    'Z:/Users/yj368/ROI isentification/SH/Control/Indiv2_test.txt',
    'Z:/Users/yj368/ROI isentification/SH/FCCP/Indiv/0.1uM.txt',
    'Z:/Users/yj368/ROI isentification/SH/FCCP/Indiv/0.5uM.txt',
    'Z:/Users/yj368/ROI isentification/SH/FCCP/Indiv/1uM.txt',
    'Z:/Users/yj368/ROI isentification/SH/FCCP/Indiv/5uM_test.txt',
]'''

'''group_name = ['Control', 'Rotenone 0.1uM', 'Rotenone 5uM', 'Rotenone 10uM']
group_paths = [
    'Z:/Users/yj368/ROI isentification/SH/Control/Indiv/Control.txt',
    'Z:/Users/yj368/ROI isentification/SH/Rotenone/Rotenone/Actin/Indiv/0.1uM.txt',
    'Z:/Users/yj368/ROI isentification/SH/Rotenone/Rotenone/Actin/Indiv/5uM.txt',
    'Z:/Users/yj368/ROI isentification/SH/Rotenone/Rotenone/Actin/Indiv/10uM_test.txt'
]'''

'''group_name = ['Control', 'Tunicamycin 0.1ug/ml', 'Tunicamycin 1ug/ml', 'Tunicamycin 5ug/ml', 'Tunicamycin 10ug/ml']
group_paths = [
    'Z:/Users/yj368/ROI isentification/SH/Control/Indiv2_test.txt',
    'Z:/Users/yj368/ROI isentification/SH/Tunicamycin/Tunicamycin/Actin/Indiv/0.1ug.txt',
    'Z:/Users/yj368/ROI isentification/SH/Tunicamycin/Tunicamycin/Actin/Indiv/1ug.txt',
    'Z:/Users/yj368/ROI isentification/SH/Tunicamycin/Tunicamycin/Actin/Indiv/5ug.txt',
    'Z:/Users/yj368/ROI isentification/SH/Tunicamycin/Tunicamycin/Actin/Indiv/10ug_test.txt'
]'''

group_name = ['Control', 'Thapsigargin 10uM', 'Thapsigargin 20uM']
group_paths = [
    'Z:/Users/yj368/ROI isentification/SH/Control/Indiv2_test.txt',
    'Z:/Users/yj368/ERstress/Indiv/Thapsigargin_10uM_test.txt',
    'Z:/Users/yj368/ERstress/Indiv/Thapsigargin_20uM_test.txt'
]

'''group_name = ['Control', 'NAC only', '500uM H2O2', '500uM H2O2 repeat', '2.5mM NAC + 500uM H2O2', '2.5mM NAC + 500uM H2O2 repeat1', '2.5mM NAC + 500uM H2O2 repeat2']
group_paths = [
    'Z:/Users/yj368/ROI isentification/SH/NAC/24012026_NAC2.5uM/Indiv/control.txt',
    'Z:/Users/yj368/ROI isentification/SH/NAC/24012026_NAC2.5uM/Indiv/NAC.txt',
    'Z:/Users/yj368/ROI isentification/SH/H2O2/correct/Indiv/500uM.txt',
    'Z:/Users/yj368/ROI isentification/SH/NAC/10022026_NAC2.5uM/Indiv/H2O2_2.txt',
    'Z:/Users/yj368/ROI isentification/SH/NAC/24012026_NAC2.5uM/Indiv/NAC_H2O2.txt',
    'Z:/Users/yj368/ROI isentification/SH/NAC/10022026_NAC2.5uM/Indiv/H2O2_NAC.txt',
    'Z:/Users/yj368/ROI isentification/SH/NAC/20012026_NAC2.5uM/Indiv_notblur.txt'
]'''

'''group_name = ['Control','2.5uM PFFs 24hrs repeat1', '2.5uM PFFs 24hrs repeat2', '2.5uM PFFs 48hrs repeat1', '2.5uM PFFs 48hrs repeat2', '2.5uM PFFs 48hrs repeat3']
group_paths = [
    'Z:/Users/yj368/ROI isentification/SH/Control/Indiv/Control.txt',
    'Z:/Users/yj368/ROI isentification/SH/PFFs/24hours/Indiv/2.5uM_repeat2.txt',
    'Z:/Users/yj368/ROI isentification/SH/PFFs/24hours/Indiv/2.5uM_repeat3.txt',
    'Z:/Users/yj368/ROI isentification/SH/PFFs/48hours/Indiv/repeat1.txt',
    'Z:/Users/yj368/ROI isentification/SH/PFFs/48hours/Indiv/repeat2.txt',
    'Z:/Users/yj368/ROI isentification/SH/PFFs/48hours/Indiv/repeat3.txt'
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

'''group_name = ['Control', 'Tunicamycin', 'Thapsigargin', 'MG132']
group_paths = [
    'Z:/Users/yj368/ERstress/Indiv/Control.txt',
    'Z:/Users/yj368/ROI isentification/SH/Tunicamycin/Tunicamycin/Actin/Indiv/10ug_test.txt',
    'Z:/Users/yj368/ERstress/Indiv/Thapsigargin.txt',
    'Z:/Users/yj368/ERstress/Indiv/MG132.txt'
]'''

n_classes = 6
class_types = ['Control', 'FCCP','H2O2', 'Rotenone','Tunicamycin', 'Thapsigargin']

class_colors = {
    'Control':   '#B8B8B8',  # light gray
    'FCCP':      '#6F8FBF',  # soft blue
    'H2O2':      '#8C79B8',  # soft purple
    'Rotenone':  '#6FAF9F',  # soft green
    'Tunicamycin': '#F0A0C0',  # soft pink
    'Thapsigargin': '#F0C080'  # soft yellow
}

all_probs = get_probabilities(model, group_paths, group_name)
pivot_df = cal_mean(all_probs, group_name, class_types)
plot_stacked_bar(pivot_df, class_types, class_colors)
plot_wasserstein_distance(pivot_df, group_name)

