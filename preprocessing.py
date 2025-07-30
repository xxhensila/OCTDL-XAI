import os  # Provides a way of using operating system-dependent functionality
from os import listdir  # Imports function to list directory contents
from os.path import isfile  # Checks if a path is a file
from os.path import join  # Joins one or more path components
from pathlib import Path  # Object-oriented filesystem paths
import cv2  # OpenCV for image processing
from math import ceil  # Rounds up to the nearest whole number
import argparse  # For parsing command-line arguments
from tqdm import tqdm  # Displays a progress bar
from sklearn.model_selection import train_test_split  # Splits arrays into train/test sets
import pandas as pd  # Library for data manipulation

# Set up command-line argument parser
parser = argparse.ArgumentParser()

# Define expected command-line arguments
parser.add_argument('--dataset_folder', type=str, default='./OCTDL_dataset', help='path to dataset folder')  # Dataset folder path
parser.add_argument('--labels_path', type=str, default='./OCTDL_dataset/labels.csv', help='path to labels.csv')  # Path to label CSV
parser.add_argument('--output_folder', type=str, default='./dataset_1', help='path to output folder')  # Output folder for processed images
parser.add_argument('--crop_ratio', type=int, default=1, help='central crop ratio of image')  # Crop ratio for center crop
parser.add_argument('--image_dim', type=int, default=512, help='final dimensions of image')  # Final size of output image
parser.add_argument('--val_ratio', type=float, default=0.15, help='validation size')  # Validation set ratio
parser.add_argument('--test_ratio', type=float, default=0.25, help='test size')  # Test set ratio
parser.add_argument('--padding', type=bool, default=False, help='padding to square')  # Whether to apply padding
parser.add_argument('--crop', type=bool, default=False, help='crop')  # Whether to crop image
parser.add_argument('--resize', type=bool, default=False, help='resize')  # Whether to resize image

# List of class labels
labels = ['AMD', 'DME', 'ERM', 'NO', 'RAO', 'RVO', 'VID']  # Disease labels
# Dataset split folders
folders = ['train', 'val', 'test']  # Folder names for data splits

# Main function
def main():
    args = parser.parse_args()  # Parse the command-line arguments
    root_folder = Path(args.dataset_folder)  # Path to input dataset folder
    output_folder = Path(args.output_folder)  # Path to output folder
    val_ratio = args.val_ratio  # Validation set ratio
    test_ratio = args.test_ratio  # Test set ratio
    train_ratio = 1 - val_ratio - test_ratio  # Compute training set ratio
    dim = (args.image_dim, args.image_dim)  # Final image dimensions
    crop_ratio = args.crop_ratio  # Crop ratio
    padding_bool = args.padding  # Whether to pad image
    crop_bool = args.crop  # Whether to crop image
    resize_bool = args.resize  # Whether to resize image
    labels_path = args.labels_path  # Path to labels CSV
    
    df = pd.read_csv(labels_path)  # Read the label file into DataFrame
    
    for folder in folders:  # Loop through train, val, test
        for label in labels:  # Loop through each label
            Path(os.path.join(output_folder, folder, label)).mkdir(parents=True, exist_ok=True)  # Create directories

    for label in tqdm(labels):  # Loop through each disease label with progress bar
        df_label = df[df['disease'] == label][['file_name', 'disease', 'patient_id']]  # Filter label-specific rows
        patients_list = df_label.patient_id.unique()  # Get unique patient IDs
        train_patients, test_patients = train_test_split(patients_list, test_size=1 - train_ratio)  # Split into train/test
        val_patients, test_patients = train_test_split(test_patients, test_size=test_ratio / (test_ratio + val_ratio))  # Split val/test
        df_label_train = df_label[df_label['patient_id'].isin(train_patients)]  # Train subset
        df_label_val = df_label[df_label['patient_id'].isin(val_patients)]  # Validation subset
        df_label_test = df_label[df_label['patient_id'].isin(test_patients)]  # Test subset
        
        print(label, len(df_label_train), len(df_label_val), len(df_label_test))  # Print sample counts
        
        for i in range(0, len(df_label_train)):  # Loop through training samples
            file_name = df_label_train.iloc[i, 0] + '.jpg'  # Construct file name
            file_label = df_label_train.iloc[i, 1]  # Get label
            preprocessing(root_folder, output_folder, file_name, 'train', crop_ratio, dim, file_label, padding_bool, crop_bool, resize_bool)  # Preprocess
            
        for i in range(0, len(df_label_test)):  # Loop through test samples
            file_name = df_label_test.iloc[i, 0] + '.jpg'  # Construct file name
            file_label = df_label_test.iloc[i, 1]  # Get label
            preprocessing(root_folder, output_folder, file_name, 'test', crop_ratio, dim, file_label, padding_bool, crop_bool, resize_bool)  # Preprocess
    
        for i in range(0, len(df_label_val)):  # Loop through validation samples
            file_name = df_label_val.iloc[i, 0] + '.jpg'  # Construct file name
            file_label = df_label_val.iloc[i, 1]  # Get label
            preprocessing(root_folder, output_folder, file_name, 'val', crop_ratio, dim, file_label, padding_bool, crop_bool, resize_bool)  # Preprocess

# Preprocessing function for individual images
def preprocessing(root_folder, output_folder, file, folder, crop_ratio, dim, label, padding_bool, crop_bool, resize_bool):
    img = cv2.imread(os.path.join(root_folder, label, file))  # Read image file
    if padding_bool:  # If padding enabled
        img = padding(img)  # Apply padding
    if crop_bool:  # If cropping enabled
        img = center_crop(img, (img.shape[1] * crop_ratio, img.shape[0] * crop_ratio))  # Apply center crop
    if resize_bool:  # If resizing enabled
        img = cv2.resize(img, dim, interpolation=cv2.INTER_LANCZOS4)  # Resize to final dimensions
    cv2.imwrite(os.path.join(output_folder, folder, label, Path(file).name), img, [cv2.IMWRITE_JPEG_QUALITY, 100])  # Save image

# Function to pad an image to square
def padding(img):
    height = img.shape[0]  # Get image height
    width = img.shape[1]  # Get image width
    if width == height:  # If already square
        return img  # Return original image
    elif width > height:  # If width is greater
        left = 0  # No left padding
        right = 0  # No right padding
        bottom = ceil((width - height) / 2)  # Bottom padding
        top = ceil((width - height) / 2)  # Top padding
        result = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))  # Add border
        return result  # Return padded image
    else:  # If height is greater
        left = ceil((height - width) / 2)  # Left padding
        right = ceil((height - width) / 2)  # Right padding
        bottom = 0  # No bottom padding
        top = 0  # No top padding
        result = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))  # Add border
        return result  # Return padded image

# Function to crop image from center
def center_crop(img, dim):
    width, height = img.shape[1], img.shape[0]  # Get width and height
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]  # Adjust crop width
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]  # Adjust crop height
    mid_x, mid_y = int(width/2), int(height/2)  # Get center of image
    cw2, ch2 = int(crop_width/2), int(crop_height/2)  # Half crop sizes
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]  # Perform cropping
    return crop_img  # Return cropped image

# Function to scale image by a factor
def scale_image(img, factor=1):
    width = int(img.shape[1] * factor)  # Compute new width
    height = int(img.shape[0] * factor)  # Compute new height
    dsize = (width, height)  # New size tuple
    output = cv2.resize(img, dsize, interpolation=cv2.INTER_LANCZOS4)  # Resize image
    return output  # Return resized image

# Entry point of the script
if __name__ == "__main__":  
    main()  # Run main function
