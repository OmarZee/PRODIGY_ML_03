import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score

# Defining image parameter
img_size = (64, 64)

# Defining directories
cat_dir = 'C:/Users/omars/OneDrive/Desktop/Work/Prodigy InfoTech/PRODIGY_ML_03/PetImages/Cats'
dog_dir = 'C:/Users/omars/OneDrive/Desktop/Work/Prodigy InfoTech/PRODIGY_ML_03/PetImages/Dogs'
train_dir = 'C:/Users/omars/OneDrive/Desktop/Work/Prodigy InfoTech/PRODIGY_ML_03/Data/Train'
test_dir = 'C:/Users/omars/OneDrive/Desktop/Work/Prodigy InfoTech/PRODIGY_ML_03/Data/Test'

# Choosing the images
cat_images = [os.path.join(cat_dir, img) for img in os.listdir(cat_dir) if img.endswith('.jpg')]
dog_images = [os.path.join(dog_dir, img) for img in os.listdir(dog_dir) if img.endswith('.jpg')]

# Splitting the data into train and test
cat_train, cat_test = train_test_split(cat_images, test_size=0.2, random_state=42)
dog_train, dog_test = train_test_split(dog_images, test_size=0.2, random_state=42)

# Copying files from dataset to train and test directories
def copy_files(file_list, dest_dir):
    for file in file_list:
        shutil.copy(file, dest_dir)

copy_files(cat_train, os.path.join(train_dir, 'Cats'))
copy_files(cat_test, os.path.join(test_dir, 'Cats'))
copy_files(dog_train, os.path.join(train_dir, 'Dogs'))
copy_files(dog_test, os.path.join(test_dir, 'Dogs'))

def preprocess_images(folder_path):
    images = []
    labels = []
    
    for label, subfolder in enumerate(['Cats', 'Dogs']):
        subfolder_path = os.path.join(folder_path, subfolder)
        for img_name in os.listdir(subfolder_path):
            img_path = os.path.join(subfolder_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                img = img.flatten()
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

X_train, Y_train = preprocess_images(train_dir)
X_test, Y_test = preprocess_images(test_dir)

X_train = X_train / 255.0
X_test = X_test / 255.0

print("Before PCA - Training data shape:", X_train.shape)
print("Before PCA - Testing data shape:", X_test.shape)

n_components = 1000
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("After PCA - Training data shape:", X_train_pca.shape)
print("After PCA - Testing data shape:", X_test_pca.shape)

svm = SVC(kernel='rbf', gamma='scale', C=1.0)
svm.fit(X_train_pca, Y_train)

Y_pred = svm.predict(X_test_pca)

print("Classification Report:\n", classification_report(Y_test, Y_pred))
print("Accuracy:", accuracy_score(Y_test, Y_pred))