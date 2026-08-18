import torch
import torch.nn as nn
import argparse
from transformer_4_4classes import ResNet50_ViT
import torchvision
from torch.utils.data import DataLoader
from data_handler_224_cold import Get_Dataset, CustomTransform, ToTensor
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

opt= argparse.Namespace()
opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
opt.train_path=None
opt.test_batch_size=1

# recall= torch.load('Z:/Users/yj368/ROI isentification/Health_state_prediction/organised training code/work for mito and lyso identification task/ResNet_mito_new_model/Stress Mechanism/transResNet_multitask_4classes_9patch_all_42du_transformer4_2.pt')
# recall= torch.load('Z:/Users/yj368/ROI isentification/Health_state_prediction/organised training code/work for mito and lyso identification task/ResNet_mito_new_model/Stress Mechanism/transResNet_multitask_4classes_9patch_all_transformer4_confocal_retain_8.pt')
# recall= torch.load('Z:/Users/yj368/ROI isentification/Health_state_prediction/organised training code/work for mito and lyso identification task/ResNet_mito_new_model/Stress Mechanism/transResNet_multitask_4classes_9patch_all_transformer4_SIM_retrain_betterreconstruction_8.pt')
recall= torch.load('Z:/Users/yj368/ROI isentification/Health_state_prediction/organised training code/work for mito and lyso identification task/ResNet_mito_new_model/Stress Mechanism/transResNet_multitask_4classes_9patch_Fixed_87_12epoch.pt')
model= ResNet50_ViT()
model.load_state_dict(recall['model'])
print(recall['best_acc'])

model.to(opt.device)
model.eval()

'''group_paths = [
    'Z:/Users/yj368/Mitochondria classification data/COS7/Control/unseen_control.txt',
    'Z:/Users/yj368/Mitochondria classification data/COS7/FCCP concentration/5uM_unseen.txt',
    'Z:/Users/yj368/Mitochondria classification data/COS7/H2O2 concentration/100mM_unseen_50img.txt',
    'Z:/Users/yj368/Mitochondria classification data/COS7/Heat/42_1h_50img.txt'
]
'''
'''group_paths = [
    'Z:/Users/yj368/ROI isentification/healthy_unhealthy_cells/SIM_retrain/Reconstructed_SIM/Control_test.txt',
    'Z:/Users/yj368/ROI isentification/healthy_unhealthy_cells/SIM_retrain/Reconstructed_SIM/FCCP_test.txt',
    'Z:/Users/yj368/ROI isentification/healthy_unhealthy_cells/SIM_retrain/Reconstructed_SIM/H2O2_test.txt',
    'Z:/Users/yj368/ROI isentification/healthy_unhealthy_cells/SIM_retrain/Reconstructed_SIM/Heatshock_test.txt'
]'''
'''group_paths = [
    'Z:/Users/yj368/ROI isentification/healthy_unhealthy_cells/SIM_retrain/Better reconstruction/Control_new_test.txt',
    'Z:/Users/yj368/ROI isentification/healthy_unhealthy_cells/SIM_retrain/Better reconstruction/FCCP_test.txt',
    'Z:/Users/yj368/ROI isentification/healthy_unhealthy_cells/SIM_retrain/Better reconstruction/H2O2_test.txt',
    'Z:/Users/yj368/ROI isentification/healthy_unhealthy_cells/SIM_retrain/Better reconstruction/Heatshock_test.txt'
]'''

'''group_paths = [
    'Z:/Users/yj368/Mitochondria classification data/Confocal/Mitochondria phenotyping/Control_all.txt',
    'Z:/Users/yj368/Mitochondria classification data/Confocal/Mitochondria phenotyping/FCCP_test.txt',
    'Z:/Users/yj368/Mitochondria classification data/Confocal/Mitochondria phenotyping/H2O2_test.txt',
    'Z:/Users/yj368/Mitochondria classification data/Confocal/Mitochondria phenotyping/Heatshock_test.txt'
]'''

group_paths = [
    'Z:/Users/yj368/ROI isentification/healthy_unhealthy_cells/Fix/Control_test.txt',
    'Z:/Users/yj368/ROI isentification/healthy_unhealthy_cells/Fix/FCCP_test.txt',
    'Z:/Users/yj368/ROI isentification/healthy_unhealthy_cells/Fix/H2O2_test.txt',
    'Z:/Users/yj368/ROI isentification/healthy_unhealthy_cells/Fix/Heat_test.txt'
]

class_types = ['Control', 'FCCP', "H\u2082O\u2082", 'Heat Shock']

y_true = []
y_pred = []

model.eval()
with torch.no_grad():
    for true_label, path in enumerate(group_paths):

        dataset = Get_Dataset(path, Transform=ToTensor())
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        for inputs, _ in loader:
            inputs = inputs.to(opt.device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            print(f'True Label: {class_types[true_label]}, Predicted Label: {class_types[preds.item()]}')

            y_true.append(true_label)
            y_pred.append(preds.item())

cm = confusion_matrix(y_true, y_pred)

'''plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="viridis",
    xticklabels=class_types,
    yticklabels=class_types
)'''

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap=sns.color_palette("light:#EB998A", as_cmap=True),
    xticklabels=class_types,
    yticklabels=class_types,
    cbar_kws={'label': 'Proportion'},
    annot_kws={"size": 14}
)

cbar = plt.gca().collections[0].colorbar
cbar.set_label('Proportion', size=14)
cbar.ax.tick_params(labelsize=14)

plt.xlabel("Predicted Class", fontsize=18)
plt.ylabel("True Class", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Confusion Matrix of Disease Mechanism Classification")
plt.tight_layout()
plt.show()



