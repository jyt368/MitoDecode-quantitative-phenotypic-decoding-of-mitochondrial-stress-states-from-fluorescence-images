# Max intensity projection for z stacks
import numpy as np
import tifffile as tiff
import os

# Load the stack image
rootdata = 'C:/path/to/your/raw/images'
output_root = 'C:/path/to/your/Max_proj/images'
for root, dirs, files in os.walk(rootdata):
        files = [f for f in files if f.endswith('.tif')]
        for i in range(len(files)):
            stack_image = tiff.imread(os.path.join(root, files[i]))
            output_image = np.max(stack_image, axis=0)

            # Get filename
            file_name = files[i]
            treated_number = file_name.split('_')[5]
            print(treated_number)
            output_path = os.path.join(output_root, f'treated_{treated_number}.tif')

            tiff.imwrite(output_path, output_image)

