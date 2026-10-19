import torch
import torch.nn as nn
import argparse
from Model_architecture import ResNet50_ViT
import torchvision
from torch.utils.data import DataLoader
#need to change to data_handler according to class number
from data_handler import Get_Dataset, CustomTransform, ToTensor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
import torch.nn.functional as F
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from openpyxl import Workbook


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
    all_probs = []

    with torch.no_grad():
        for i, group in enumerate(group_paths):
            dataset = Get_Dataset(group, Transform=ToTensor())
            loader = DataLoader(dataset, batch_size=1, shuffle=False)

            for inputs, labels in loader:
                inputs = inputs.to(opt.device)

                output = model(inputs)
                probs = F.softmax(output, dim=1).cpu().numpy()[0]

                all_probs.append({
                    'group': group_name[i],
                    'p1': probs[0],
                    'p2': probs[1],
                    'p3': probs[2],
                    'p4': probs[3],
                })
    return all_probs

def bar_plot_probabilities(all_probs, group_name, class_colors=None, points_colors=None): 
    df = pd.DataFrame(all_probs)
    df_long = df.melt(
        id_vars='group',
        value_vars=['p1', 'p2', 'p3', 'p4'],
        var_name='class',
        value_name='probability'
    )
    df_long['probability'] = df_long['probability'] * 100
    
    # Save probabilities
    output_path = "Z:/Users/yj368/Mitochondria classification data/COS7/class_probabilities_summary.xlsx"
    df_long.to_excel(output_path, index=False)
    print(f"Saved summary to {output_path}")

    # Mean and SEM bars
    summary = df_long.groupby(['group', 'class']).agg(
        mean_prob=('probability', 'mean'),
        sem_prob=('probability', stats.sem)
    ).reset_index()

    plt.figure(figsize=(9, 4))
    groups = summary['group'].unique()
    classes = ['p1', 'p2', 'p3', 'p4']

    x = np.arange(len(groups))
    width = 0.18
    class_labels = {
        'p1': 'Control',
        'p2': 'FCCP',
        'p3': 'H2O2',
        'p4': 'Heat Shock'
    }

    for i, cls in enumerate(classes):
        cls_data = summary[summary['class'] == cls]
        plt.bar(
            x + i * width,
            cls_data['mean_prob'],
            width=width,
            yerr=cls_data['sem_prob'],
            capsize=4,
            label=class_labels[cls],
            color=class_colors[cls],
            edgecolor='black',
            linewidth=1,
            zorder=2,
            error_kw={'ecolor': points_colors[cls]}
        )

        for j, group in enumerate(groups):
            group_data = df_long[
                (df_long['group'] == group) & (df_long['class'] == cls)
            ]

            jitter = np.random.normal(0, 0.015, size=len(group_data))

            plt.scatter(
                np.full(len(group_data), x[j] + i * width) + jitter,
                group_data['probability'],
                color=points_colors[cls],
                linewidth=0.3,
                alpha=0.8,
                s=5,
                zorder=3
            )

    ax = plt.gca()
    ax.set_xticks(x + 1.5 * width)
    plt.xticks(x + 1.5 * width, groups, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Phenotypic probability \n distribution(%)', fontsize=12)
    plt.xlabel('Treatment', fontsize=12)

    plt.ylim(0, 100) 

    sns.despine()
    plt.legend(title='Stress type', frameon=False, loc='center left',bbox_to_anchor=(1.02, 0.5), fontsize=12, title_fontsize=12)

    plt.tight_layout()
    plt.show()

'''---main---'''
opt = load_opt()
pth= 'Z:/Users/yj368/ROI isentification/Health_state_prediction/organised training code/work for mito and lyso identification task/Code_for_publication/trained_models/COS7_widefield.pt'
model=load_model(pth)

group_name = ['Control', 'FCCP', "H2O2", 'Heat Shock']
group_paths = [
    'Z:/Users/yj368/Mitochondria classification data/COS7/Control/unseen_control.txt',
    'Z:/Users/yj368/Mitochondria classification data/COS7/FCCP_concentration/5uM_unseen.txt',
    'Z:/Users/yj368/Mitochondria classification data/COS7/H2O2_concentration/100mM_unseen_50img.txt',
    'Z:/Users/yj368/Mitochondria classification data/COS7/Heat/42_1h_50img.txt'
]

'''group_name = ['Control', 'Cold_RT','Heat Shock 38']
group_paths = [
    'Z:/Users/yj368/Mitochondria classification data/COS7/Control/unseen_control.txt',
    'Z:/Users/yj368/Mitochondria classification data/COS7/Cold/RT/RT_all.txt',
    'Z:/Users/yj368/Mitochondria classification data/COS7/Heat/38_all.txt'
]'''


class_colors = {
    'p1': '#B8B8B8',  
    'p2': '#6F8FBF',  
    'p3': '#8C79B8',  
    'p4': '#F0C060',
}

points_colors = {
    'p1': "#817E7E",  
    'p2': "#2C69C5",  
    'p3': "#754FCC",  
    'p4': "#EC8D11",
}

all_probs = get_probabilities(model, group_paths, group_name)
bar_plot_probabilities(all_probs, group_name, class_colors, points_colors)