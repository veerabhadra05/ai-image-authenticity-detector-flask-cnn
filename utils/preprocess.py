import cv2
import numpy as np
import os

def preprocess_image(image_path, target_size=(224, 224)):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img.astype('float32') / 255.0
        
        return img
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def load_dataset(dataset_path, target_size=(224, 224)):
    """
    Load images and labels from the dataset directory.
    FAKE = 0, REAL = 1
    """
    images = []
    labels = []
    
    categories = {'FAKE': 0, 'REAL': 1}
    
    for category, label in categories.items():
        category_path = os.path.join(dataset_path, category)
        if not os.path.exists(category_path):
            continue
            
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = preprocess_image(img_path, target_size)
            
            if img is not None:
                images.append(img)
                labels.append(label)
                
    return np.array(images), np.array(labels)
