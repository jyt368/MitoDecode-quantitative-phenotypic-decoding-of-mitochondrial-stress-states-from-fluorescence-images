import torch
import torch.nn as nn
import argparse
from transformer_4_4classes import ResNet50_ViT
import torchvision
from torch.utils.data import DataLoader
#need to change to data_handler_224_mito_inf when testing infinitus compounds; else, use data_handler_224_mito_plot
from data_handler_224 import Get_Dataset, CustomTransform, ToTensor
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

                all_probs.append({
                    'group': group_name[i],
                    'p1': probs[0],
                    'p2': probs[1],
                    'p3': probs[2],
                    'p4': probs[3],
                })
    return all_probs

def confidence_ellipse(points, ax, color,
                       conf=0.90,  # 90% confidence interval
                       fill_alpha=0.50,
                       edge_alpha=0.5,
                       linewidth=2.2):

    cov = np.cov(points.T)
    mean = points.mean(axis=0)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    angle = np.degrees(np.arctan2(*eigvecs[:,0][::-1]))

    scale = np.sqrt(chi2.ppf(conf, df=2))

    width = 2 * scale * np.sqrt(eigvals[0])
    height = 2 * scale * np.sqrt(eigvals[1])

    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        lw=linewidth,
        alpha=fill_alpha,
        zorder=1
    )

    ax.add_patch(ellipse)

     # Border
    ellipse_edge = Ellipse(
        mean,
        width,
        height,
        angle=angle,
        facecolor='none',
        edgecolor=color,
        linewidth=linewidth,
        alpha=edge_alpha,
        zorder=4
    )
    ax.add_patch(ellipse_edge)

from scipy.spatial.distance import mahalanobis

def remove_outliers(points, conf=0.90):

    mean = points.mean(axis=0)
    cov = np.cov(points.T)

    inv_cov = np.linalg.inv(cov)

    d = np.array([
        mahalanobis(p, mean, inv_cov)
        for p in points
    ])

    threshold = np.sqrt(chi2.ppf(conf, df=2))

    return points[d <= threshold]

def jitter_points(points, scale=0.015, seed=40):
    rng = np.random.default_rng(seed)

    # scale based on PCA range
    spread = max(
        np.ptp(points[:,0]),
        np.ptp(points[:,1])
    )

    jitter = scale * spread

    x = points[:,0] + rng.normal(0, jitter, len(points))
    y = points[:,1] + rng.normal(0, jitter, len(points))

    return np.column_stack([x, y])

def plot_pca(all_probs, group_name, class_colors=None):    
    df = pd.DataFrame(all_probs)
    print(df.head())

    X = df[['p1','p2','p3','p4']].values

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    df['PC1'] = X_pca[:, 0]
    df['PC2'] = X_pca[:, 1]

    explained_var = pca.explained_variance_ratio_

    sns.set(style="white", context="talk", font_scale=1.1)

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    for g in group_name:
        subset = df[df['group'] == g]
        points = subset[['PC1', 'PC2']].values
        color = class_colors[g]

        ellipse_points = remove_outliers(points, conf=0.90)

        if len(ellipse_points) > 5:
            confidence_ellipse(
                ellipse_points,
                ax,
                color=color,
                conf=0.90
            )

        # scatter
        plot_points = jitter_points(points, scale=0.015)
        plt.scatter(
            plot_points[:,0],
            plot_points[:,1],
            label=g,
            s=30,
            alpha=0.75,
            color=color,
            edgecolor='black',
            linewidth=0.5,
            zorder=2
        )

        centroid = points.mean(axis=0)
        plt.scatter(
            centroid[0],
            centroid[1],
            color=color,
            marker='X',
            s=100,
            edgecolor='black',
            linewidth=1,
            zorder=4
        )

    plt.xlabel(f'PC1 ({explained_var[0]*100:.1f}% variance)')
    plt.ylabel(f'PC2 ({explained_var[1]*100:.1f}% variance)')
    plt.legend(frameon=False)
    sns.despine()
    plt.title('PCA of Model-Derived Probability Vectors')
    plt.tight_layout()
    plt.show()


'''---main---'''
opt = load_opt()
pth= 'Z:/Users/yj368/ROI isentification/Health_state_prediction/organised training code/work for mito and lyso identification task/ResNet_mito_new_model/Stress Mechanism/transResNet_multitask_4classes_9patch_all_42du_transformer4_2.pt'
model=load_model(pth)

group_name = ['Control', 'FCCP', "H2O2", 'Heat Shock']
group_paths = [
    'Z:/Users/yj368/Mitochondria classification data/COS7/Control/unseen_control.txt',
    'Z:/Users/yj368/Mitochondria classification data/COS7/FCCP concentration/5uM_unseen.txt',
    'Z:/Users/yj368/Mitochondria classification data/COS7/H2O2 concentration/100mM_unseen_50img.txt',
    'Z:/Users/yj368/Mitochondria classification data/COS7/Heat/42_1h_50img.txt'
]

'''group_name = ['Control', 'RT', 'Heat shock 38°C']
group_paths = [
    'Z:/Users/yj368/ROI isentification/healthy_unhealthy_cells/data/FCCP concentration/0.5uM/test_control.txt',
    'Z:/Users/yj368/Cold/SIM0707_cold/widefield/RT_1hour.txt',
    'Z:/Users/yj368/ROI isentification/healthy_unhealthy_cells/Heat/38_1h.txt'
]'''

'''class_colors = {
    'Control':   '#B8B8B8',  # light gray
    'RT':      '#6F8FBF',  # soft blue
    'H2O2':      '#8C79B8',  # soft purple
    'Heat shock 38°C':'#F0C060',  # yellow
}'''

class_colors = {
    'Control':   '#B8B8B8',  # light gray
    'FCCP':      '#6F8FBF',  # soft blue
    'H2O2':      '#8C79B8',  # soft purple
    'Heat Shock':'#F0C060',  # yellow
}

all_probs = get_probabilities(model, group_paths, group_name)
plot_pca(all_probs, group_name, class_colors)