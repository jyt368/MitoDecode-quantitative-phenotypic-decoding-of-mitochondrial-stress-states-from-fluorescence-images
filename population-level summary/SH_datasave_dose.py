import sys
sys.path.insert(1, 'C:/code_download_path/MitoDecode - main')

import torch
import torch.nn as nn
import argparse
from main_model.Model_architecture import ResNet50_ViT
import torchvision
from torch.utils.data import DataLoader

from main_model.data_handler import Get_Dataset, CustomTransform, ToTensor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch.nn.functional as F

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
                data.append({
                    'group': group_name[i],
                    'p1': probs[0,0].item() * 100,
                    'p2': probs[0,1].item() * 100,
                    'p3': probs[0,2].item() * 100,
                    'p4': probs[0,3].item() * 100
                })

    pack=pd.DataFrame(data)
    # print(pack.head())

    return pack

'''------------------- Main ------------------'''
opt = load_opt()
pth= 'C:/saved_model_path/SH_4classes.pt'
model=load_model(pth)

'''group_name = ['Control', 'H2O2 10uM', 'H2O2 50uM', 'H2O2 100uM', 'H2O2 500uM', 'H2O2 1mM']
group_paths = [
    'Z:/path/to/Mitochondria_classification_data/SHSY5Y/Control/Indiv2/Control_test_all.txt',
    'Z:/path/to/Mitochondria_classification_data/SHSY5Y/H2O2/10uM.txt',
    'Z:/path/to/Mitochondria_classification_data/SHSY5Y/H2O2/50uM.txt',
    'Z:/path/to/Mitochondria_classification_data/SHSY5Y/H2O2/100uM.txt',
    'Z:/path/to/Mitochondria_classification_data/SHSY5Y/H2O2/500uM.txt',
    'Z:/path/to/Mitochondria_classification_data/SHSY5Y/H2O2/1mM_test.txt'
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
    'Z:/path/to/Mitochondria_classification_data/SHSY5Y/Control/Indiv2/Control_test_all.txt',
    'Z:/path/to/Mitochondria_classification_data/SHSY5Y/Rotenone/0.1uM.txt',
    'Z:/path/to/Mitochondria_classification_data/SHSY5Y/Rotenone/5uM.txt',
    'Z:/path/to/Mitochondria_classification_data/SHSY5Y/Rotenone/10uM_test.txt'
]


'''group_name = ['Control', 'NAC only', '500uM H2O2', '2.5mM NAC + 500uM H2O2']
group_paths = [
    'Z:/path/to/Mitochondria_classification_data/SHSY5Y/NAC/control.txt',
    'Z:/path/to/Mitochondria_classification_data/SHSY5Y/NAC/NAC.txt',
    'Z:/path/to/Mitochondria_classification_data/SHSY5Y/NAC/H2O2.txt',
    'Z:/path/to/Mitochondria_classification_data/SHSY5Y/NAC/NAC_H2O2_all.txt'
]'''

'''group_name = ['Control','2.5uM PFFs 24 hours', '2.5uM PFFs 48hours']
group_paths = [
    'Z:/path/to/Mitochondria_classification_data/SHSY5Y/Control/Indiv2/Control_test_all.txt',
    'Z:/path/to/Mitochondria_classification_data/SHSY5Y/PFFs/24hour_all.txt',
    'Z:/path/to/Mitochondria_classification_data/SHSY5Y/PFFs/48hour_all.txt'
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
export_df = pack[['group', 'p4']]
export_df.to_excel("C:/save_path/Rotenone_probability_control.xlsx", index=False)
