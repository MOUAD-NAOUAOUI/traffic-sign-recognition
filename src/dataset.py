"""
GTSRB Dataset Loader, Preprocessing, and Augmentation Pipeline.
Job: The Chef / Food Prep Station.
Takes raw messy image files, cleans them, enhances contrast, resizes to 48x48, and serves them in batches.
"""

import os 
from pathlib import Path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf 

IMG_SIZE=(48,48)
NUM_CLASSES=43

def preprocess_image(img_bgr):
    img_rgb=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
    img_resized=cv2.resize(img_rgb,IMG_SIZE)
    lab=cv2.cvtColor(img_resized,cv2.COLOR_RGB2LAB)
    l,a,b=cv2.split(lab)
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4))
    l=clahe.apply(l)
    img_enhanced=cv2.cvtColor(cv2.merge((l,a,b)),cv2.COLOR_LAB2RGB)
    return img_enhanced.astype(np.float32)/255.0

def load_training_data(data_dir, max_per_class=None):
    train_dir = Path(data_dir) / "GTSRB_Final_Training_Images" / "GTSRB" / "Final_Training" / "Images"
    images = []
    labels = []
    for class_id in range(NUM_CLASSES):
        folder = train_dir / f"{class_id:05d}"
        if not folder.exists():
            continue
        files=[f for f in folder.iterdir() if f.suffix.lower() in ['.jpg','.ppm','.png']]
        if max_per_class:
            files = files[:max_per_class]
        for img_file in files:
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            images.append(preprocess_image(img))
            labels.append(class_id)
        print(f"Loaded {len(files)} images for class {class_id:02d}")

    X=np.array(images,dtype=np.float32)
    y=np.array(labels,dtype=np.int32)
    print(f"Total images loaded:{len(X)} with shape {X.shape}")
    X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y,)
    return X_train,X_val,y_train,y_val
def create_data_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(factor=0.05),
        tf.keras.layers.RandomZoom(height_factor=(-0.1,0.1)),
        tf.keras.layers.RandomTranslation(height_factor=0.08,width_factor=0.08),
    ],
    name="data_augmentation",
    )