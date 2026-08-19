# MitoDecode-quantitative-phenotypic-decoding-of-mitochondrial-stress-states-from-fluorescence-images

MitoDecode take fluorescence mitochondria images of individual cell directly as input and does not require individual mitochondria segmentation or predefined mitochondrial morphological parameters for analysis. It gives individual cell a health/stress probability profile to indicate the health state, stress type, and stress severity of a cell.

1. Image preprocessing

For single-cell analysis, images are first processed to generate a individual cell mitochondria data. For raw data acquired as z-stacks, maximum intensity projection is used along the z-axis to generate two-dimensional mitochondrial images. Individual cells are then segmented using a retrained Cellpose model based on specific cell types. The segmented cell masks are used to separate individual cells and generate single-cell image as inputs for MitoDecode. Here, Cellpose segmentation is used to identify individual cells rather than individual mitochondria. No mitochondrial object segmentation, skeletonisation, circularity calculation, aspect-ratio measurement or manual morphological feature selection is needed in MitoDecode analysis.

<img width="708" height="444" alt="image" src="https://github.com/user-attachments/assets/1a461797-a33d-4670-8a78-3ac972fc381d" />


<img width="797" height="540" alt="image" src="https://github.com/user-attachments/assets/2d846b99-47ab-4336-b749-ad6f52fe54d8" />


2. 
