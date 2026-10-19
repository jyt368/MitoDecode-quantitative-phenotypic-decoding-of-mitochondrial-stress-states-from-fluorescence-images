import torch
torch.cuda.empty_cache()
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
from  PIL import  Image
import numpy as np
import tifffile as tiff
import cv2
from scipy import stats

def mode_denoise(image):
    # Find the most common intensity value (background)
    mode_intensity = stats.mode(image, axis=None)[0]

    # Signals masks
    mask = image > mode_intensity
    # Remove the background
    image_no_background = np.zeros_like(image)
    image_no_background[mask] = image[mask]
    
    image_no_background[mask] = image_no_background[mask] - mode_intensity

    denoised_image = cv2.GaussianBlur(image_no_background, (5, 5), 0)

    return denoised_image

# Because the segmented image shape would be random rectangular, so we need padding to make it square and fixed size
# Default 2D images
def pad_image(img):
    target_height, target_width = max(img.shape), max(img.shape)
    height, width = img.shape[:2]
    pad_height = max(0, target_height - height)
    pad_width = max(0, target_width - width)
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    padded_img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    return padded_img

def map_labels_to_classes(label):
    label = float(label)
    if label == 0:
        return 0  # Control
    elif label == 5:
        return 1  # FCCP
    elif label == 3:
        return 2  #H2O2
    elif label == 4:
        return 3  #Heat shock
    else:
        return 4

def normalise(img, threshold=0.01,percentage=0.02):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        for c in range(img.shape[0]):
            img[c] = np.clip(img[c], 0, np.percentile(img[c], (100 - percentage)))
            img[c] = img[c] - np.percentile(img[c], percentage)
            img[c][img[c] < 0] = 0
            img[c] = img[c] / np.amax(img[c])
            img[c][img[c] < threshold] = 0
        return img.squeeze(0)

def patch(img, patch_size=224, stride=224):
        patches = []
        img_height, img_width = img.shape[0], img.shape[1]
        for i in range(0, img_height, stride):
            for j in range(0, img_width, stride):
                patch = img[i:i+patch_size, j:j+patch_size]
                patches.append(patch)  # patches shape: (224, 224)
        patches = np.array(patches) # shape: (16, 224, 224)
        # print('patches',patches.shape)
        return patches

# Data augmentation
class CustomTransform:
    def __init__(self, p_filp=0.5, rot=45, p_jitter=0.1, noise=0.01):
        self.p_filp = p_filp
        self.rot = rot
        self.p_jitter = p_jitter
        self.noise = noise    

    def __call__(self, sample):
        img_pth, label = sample
        img = tiff.imread(img_pth)
        
        #if grey scale, copy channels
        if len(img.shape) == 2:
            img = img.astype(np.float32)
            #print(img.shape)
        else: 
            img = img[1].astype(np.float32) #mito


        img = mode_denoise(img)

        img = pad_image(img)
        img = normalise(img)

        # img = normalise(img)
        # print(img.shape)

        img = np.nan_to_num(img)  # Replace NaNs with 0 and infs with large finite numbers
        img = np.clip(img, 0, 1).astype(np.float32)

        img= Image.fromarray((img * 255/np.max(img)).astype(np.uint8))  # to PIL image
        
        transform = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(p=self.p_filp), # flip probability
        transforms.RandomVerticalFlip(p=self.p_filp),
        transforms.RandomRotation(degrees=self.rot),
        transforms.ColorJitter(brightness=self.p_jitter, contrast=self.p_jitter),
        transforms.ToTensor()
        ])

        img = transform(img)
        # print(img) # tensor
        
        return img, label
    
class ToTensor:
    # Convert ndarrays to Tensors
    def __call__(self, sample):
        img_pth, label = sample
        img = tiff.imread(img_pth)
        print(img.shape)
        #if grey scale, copy channels
        if len(img.shape) == 2:
            img = img.astype(np.float32)
            #print(img.shape)
        else: 
            img = img[1].astype(np.float32) #mito channel, if I have three channels
        '''plt.imshow(img, cmap='gray')
        #combine name of the image and label as image title
        plt.title(f'Original Image: {img_pth} - Label: {img.shape}')
        plt.axis('off')
        plt.show()'''

        img = mode_denoise(img)
        img = pad_image(img)
        img = normalise(img).astype(np.float32)

        img = np.nan_to_num(img)  # Replace NaNs with 0 and infs with large finite numbers
        img = np.clip(img, 0, 1).astype(np.float32)

        img= Image.fromarray((img * 255/np.max(img)).astype(np.uint8))
        transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
        ])
        img = transform(img)

        return img, label
    
class Get_Dataset(Dataset):
    def __init__(self,input_path, Transform=None):
        self.img_paths, self.labels =self.get_images(input_path)
        self.labels = [map_labels_to_classes(label) for label in self.labels]
        # print(self.img_paths)
        # print(self.labels)
        self.Transform=Transform

    def get_images(self, input_path):
        with open(input_path,'r',encoding='UTF-8') as f:
            imgs_info=f.readlines()
            # convert '\' into '\\'-->readable for python
            imgs_info=list(map(lambda x:x.strip().split('\t'),imgs_info))
        
        # Separate image paths and labels
        img_paths = [info[0] for info in imgs_info]
        labels = [info[1] for info in imgs_info]  
        
        return img_paths, labels

    def patch(self, img, patch_size=224, stride=224):
        patches = []
        img_height, img_width = img.shape[0], img.shape[1]
        for i in range(0, img_height, stride):
            for j in range(0, img_width, stride):
                patch = img[i:i+patch_size, j:j+patch_size]
                patches.append(patch)  # patches shape: (224, 224)
        patches = np.array(patches) # shape: (16, 224, 224)
        # print('patches',patches.shape)
        return patches
    
    def __getitem__(self, index):
        path =self.img_paths[index]
        label=self.labels[index]
        sample= path,label

        if self.Transform:
            sample=self.Transform(sample)

        return sample

    def __len__(self):
        length=len(self.img_paths)
        return length

# load data with batch
def LoadData(img_path,transform, category, opt):
    dataset=Get_Dataset(img_path,Transform=transform)
    if category == 'train':
        train_loader = DataLoader(dataset=dataset,
                          batch_size=opt.train_batch_size,
                          shuffle=True,
                          num_workers=4)
        return train_loader
    else:
        test_loader = DataLoader(dataset=dataset,
                          batch_size=opt.test_batch_size,
                          shuffle=False,
                          num_workers=0)
        return test_loader
    
def Get_Data(opt):
    train_path = opt.train_path
    #validate_path = opt.validate_path
    test_path = opt.test_path

    train_transform = CustomTransform()
    #validate_transform = ToTensor()
    test_transform = ToTensor()

    train_loader = LoadData(train_path, train_transform, 'train', opt)
    #validate_loader = LoadData(validate_path, validate_transform, 'validate', opt)
    test_loader = LoadData(test_path, test_transform, 'test', opt)
    return train_loader, test_loader



