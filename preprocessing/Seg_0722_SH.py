import numpy as np
import cv2
import os
import tifffile as tiff
from cellpose_image_based_SH import Cellpose_class
import torch
import matplotlib.pyplot as plt

def normalise_channels(image):
    image = image.astype(np.float32)
    for i in range(image.shape[0]):  # Iterate over channels
        channel = image[i]
        image[i] = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
        
        plt.figure()
        plt.imshow(image[i], cmap='gray')
        plt.title(f"Channel {i + 1} (Normalized)")
        plt.axis('off')
        plt.show()
    
    img_16bit = (image * 65535).astype(np.uint16)
    return img_16bit

def segment_cells(cell_masks):
    cell_rectangles = []
    for mask in cell_masks:
        min_x, min_y = np.min(np.where(mask), axis=1)
        max_x, max_y = np.max(np.where(mask), axis=1)
        cell_rectangles.append((min_x, min_y, max_x, max_y))
    return cell_rectangles

#load my trained model pth for segmentation
def segment_cellpose(img):
    normalise_channels(img)
    img = np.transpose(img, (1, 2, 0))
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    Seg = Cellpose_class()
    Masks=Seg.generate_mask(img)
    Cell_area=Seg.count_cells(Masks)

    print(f"Number of cells detected: {Masks.max()}")

    # segment individual cells
    #rectangles = []
    cell_id = 1
    Profile = {} # profile for each FOV
    for label in range(1, Masks.max() + 1):
        binary_mask = (Masks == label).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Number of ROIs found: {len(contours)}")
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            #rectangles.append((x, y, x + w, y + h))
            cell_location = (x, y, x + w, y + h)
            Profile[cell_id] = {
                'cell_id': cell_id,
                'location': cell_location
            }
            cell_id += 1
    
    return Profile

def seg_img(img, img_pth):
    Profile = segment_cellpose(img)
    file_name = os.path.basename(input_pth)
    print(file_name)
    # image is the input FOV, and I want to segment each cell into a new image based on the rectangles, and save and name it according to the cell id
    for cell_id, location in Profile.items():
        x1, y1, x2, y2 = location['location']
        #keep all the channels
        cell_img = img[:, y1:y2, x1:x2]

        image_size_kb = cell_img.nbytes / 1024  # kB
        print(f"Cell {cell_id} size: {image_size_kb:.2f} KB")
        if image_size_kb < 300:
            print(f"Skipping cell {cell_id} as its size is {image_size_kb:.2f} KB (less than 300 kB).")
            continue
        
        filename = f"{file_name.split('.')[0]}_cell_{cell_id}.tif"
        print(f"Saving cell image: {filename}")

        tiff.imwrite(os.path.join(img_pth, filename), cell_img)
    
    print("Segmentation of this FOV completed and images saved.")

def get_images(input_path):
    with open(input_path,'r',encoding='UTF-8') as f:
        imgs_info=f.readlines()
        # convert '\' into '\\'-->readable for python
        img_paths=list(map(lambda x:x.strip().split('\t'),imgs_info)) 
    
    return img_paths

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

img_list ='Z:/Users/yj368/ERstress/Maxproj/Thapsigargin_20uM.txt'
sav_pth = 'Z:/Users/yj368/ERstress/Indiv/Thapsigargin_20uM'
img_paths = get_images(img_list)
for input_pth in img_paths:
    input_pth = input_pth[0]    
    print(input_pth)

    img = tiff.imread(input_pth)
    
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
        img = np.repeat(img, 3, axis=0)
    print(img.shape)    

    seg_img(img, sav_pth)  
