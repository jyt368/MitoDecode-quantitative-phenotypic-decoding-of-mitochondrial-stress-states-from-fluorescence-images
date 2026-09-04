# MitoDecode: quantitative phenotypic decoding of mitochondrial stress states from fluorescence images

MitoDecode take fluorescence mitochondria images of individual cell directly as input and does not require individual mitochondria segmentation or predefined mitochondrial morphological parameters for analysis. It gives individual cell a health/stress probability profile to indicate the health state, stress type, and stress severity of a cell.

  <img width="894" height="368" alt="image" src="https://github.com/user-attachments/assets/572e7e59-800a-44ad-91fd-ed4a00b51370" />

1. Image preprocessing

For single-cell analysis, images are first processed to generate a individual cell mitochondria data. For raw data acquired as z-stacks, maximum intensity projection is used along the z-axis to generate two-dimensional mitochondrial images. Individual cells are then segmented using a retrained Cellpose model based on specific cell types. The segmented cell masks are used to separate individual cells and generate single-cell image as inputs for MitoDecode. Here, Cellpose segmentation is used to identify individual cells rather than individual mitochondria. No mitochondrial object segmentation, skeletonisation, circularity calculation, aspect-ratio measurement or manual morphological feature selection is needed in MitoDecode analysis.

2. Model architecture

We designed a hybrid deep learning architecture that combines ResNet50 backbone with a Transformer encoder. This architecture took advantages of convolutional layers to extract diverse features maps from images, as well as the Transformer self-attention part to captures spatial relationships of overall mitochondria network. The feature representation from ResNet50 backbone is converted into image patches and processed by positional encoding and Transformer encoder layers to capture global morphological relationships across the cell. The final representation is passed through a fully connected (FC) layer and converted into a phenotypic probability profile. Each dimension corresponds to one specific stress phenotype represented in the training reference space.

<img width="960" height="272" alt="image" src="https://github.com/user-attachments/assets/8bd28600-4f2f-4fae-a20c-3ac9fef4317b" />


backbone: ResNet50 <br>
transformer_layers: 4 <br>
num_classes: 4 / 6 (non-mitochondrial stressors) <br>
Optimizer: AdamW <br>
Loss: Cross-entropy loss with label smoothing = 0.1 <br>
Learning-rate scheduler: StepLR <br>
learning_rate: 1e-4 <br>
batch_size: 16 <br>
epochs: 20

3. Model training and fine-tuning
The initial MitoDecode model was trained using COS-7 mitochondrial fluorescence images under four different biological states: control, FCCP-induced stress, H₂O₂-induced stress and heat-shock stress. Cells with abnormal imaging artefacts were excluded.

MitoDecode can be adapted to new cell types, stress types, and imaging modalities through supervised fine-tuning. Models were fine-tuned using small number of new condition images based on the initially trained model rather than training from scratch.

4. Installations
Python 3.12.1
The environmental setting are included as requirements.txt
run:  python GUI.py  to launch the interface for example single-cell prediction

6. Example usage
<img width="1204" height="828" alt="image" src="https://github.com/user-attachments/assets/0356ecb6-ff7c-42e1-854f-e95f5f0bf047" />


