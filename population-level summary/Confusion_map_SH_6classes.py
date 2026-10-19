import torch
import torch.nn as nn
import argparse
from transformer_4_6classes import ResNet50_ViT
import torchvision
from torch.utils.data import DataLoader
from data_handler import Get_Dataset, CustomTransform, ToTensor
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

opt= argparse.Namespace()
opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
opt.train_path=None
opt.test_batch_size=1

recall= torch.load('C:/saved_model_path/SH_6classes.pt')
model= ResNet50_ViT()
model.load_state_dict(recall)

model.to(opt.device)
model.eval()

group_paths = [
    'Z:/Users/yj368/Mitochondria classification data/SHSY5Y/Control/Indiv2/Control_test_all.txt',
    'Z:/Users/yj368/ROI isentification/SH/FCCP/Indiv/5uM_test.txt',
    'Z:/Users/yj368/ROI isentification/SH/H2O2/correct/Indiv/1mM_test.txt',
    'Z:/Users/yj368/ROI isentification/SH/Rotenone/Rotenone/Actin/Indiv/50uM_test.txt',
    'Z:/Users/yj368/Mitochondria classification data/SHSY5Y/Tunicamycin/10ug_test2.txt',
    'Z:/Users/yj368/ERstress/Indiv/Thapsigargin_20uM_test.txt'
]

class_types = ['Control', 'FCCP', "H\u2082O\u2082", 'Rotenone', 'Tunicamycin', 'Thapsigargin']

y_true = []
y_pred = []
y_prob = []

model.eval()
with torch.no_grad():
    for true_label, path in enumerate(group_paths):

        dataset = Get_Dataset(path, Transform=ToTensor())
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        for inputs, _ in loader:
            inputs = inputs.to(opt.device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
            print(f'True Label: {class_types[true_label]}, Predicted Label: {class_types[preds.item()]}')

            y_true.append(true_label)
            y_pred.append(preds.item())
            y_prob.append(probs)

cm = confusion_matrix(y_true, y_pred)

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap=sns.color_palette("light:#EB998A", as_cmap=True),
    xticklabels=class_types,
    yticklabels=class_types,
    cbar_kws={'label': 'Proportion'},
    annot_kws={"size": 12}
)

cbar = plt.gca().collections[0].colorbar
cbar.set_label('Proportion', size=12)
cbar.ax.tick_params(labelsize=12)

plt.xlabel("Predicted Class", fontsize=12)
plt.ylabel("True Class", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Confusion Matrix of Disease Mechanism Classification")
plt.tight_layout()
plt.show()

