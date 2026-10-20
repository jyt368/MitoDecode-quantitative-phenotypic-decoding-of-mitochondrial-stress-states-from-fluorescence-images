import sys
sys.path.insert(1, 'C:/code_download_path/MitoDecode - main')

import torch
import torch.nn as nn
import argparse
from main_model.Model_architecture import ResNet50_ViT   # make sure the Model_architecture number of classes have been changed to 6.
import torchvision
from torch.utils.data import DataLoader
from main_model.data_handler import Get_Dataset, CustomTransform, ToTensor
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
    model.load_state_dict(recall)
    #print(recall['best_acc'])
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
    plt.ylabel('Mean Predicted Probability (%)')
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

    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)

    sns.despine()
    plt.tight_layout()
    plt.show()

'''---main---'''
opt = load_opt()
pth= 'C:/saved_model_path/SH_6classes.pt'
model=load_model(pth)


'''group_name = ['Control', 'Tunicamycin 0.1ug/ml', 'Tunicamycin 1ug/ml', 'Tunicamycin 5ug/ml', 'Tunicamycin 10ug/ml']
group_paths = [
    'Z:/path/to/Mitochondria_classification_data/SHSY5Y/Control/Indiv2/Control_test_all.txt',
    'Z:/path/to/Mitochondria_classification_data/SHSY5Y/Tunicamycin/0.1ug.txt',
    'Z:/path/to/Mitochondria_classification_data/SHSY5Y/Tunicamycin/1ug.txt',
    'Z:/path/to/Mitochondria_classification_data/SHSY5Y/Tunicamycin/5ug.txt',
    'Z:/path/to/Mitochondria_classification_data/SHSY5Y/Tunicamycin/10ug_test2.txt'
]'''

group_name = ['Control', 'Thapsigargin 10uM', 'Thapsigargin 20uM']
group_paths = [
    'Z:/path/to/Mitochondria_classification_data/SHSY5Y/Control/Indiv2/Control_test_all.txt',
    'Z:/path/to/Mitochondria_classification_data/SHSY5Y/Thapsigargin/Thapsigargin_10uM_test.txt',
    'Z:/path/to/Mitochondria_classification_data/SHSY5Y/Thapsigargin/Thapsigargin_20uM_test.txt'
]


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
