import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# data_dir = 'C:/Users/omars/OneDrive/Desktop/Work/Prodigy InfoTech/PRODIGY_ML_03/PetImages'

# categories = ['Cat', 'Dog']

# data = []

# for category in categories:
#     path = os.path.join(data_dir, category)
#     label = categories.index(category)
#     for img in os.listdir(path):
#         imgpath = os.path.join(path, img)
#         pet_img = cv2.imread(imgpath, 0)
#         try:
#             pet_img = cv2.resize(pet_img, (50, 50))
#             image = np.array(pet_img).flatten()
#             data.append([image, label])
#         except Exception as e:
#             pass
        

# pick_in = open('data1.pickle', 'wb')
# pickle.dump(data, pick_in)
# pick_in.close()

pick_in = open('data1.pickle', 'rb')
data = pickle.load(pick_in)
pick_in.close()

random.shuffle(data)
features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.01)

# model = SVC(C=1, kernel= 'poly', gamma= 'auto')
# model.fit(X_train, Y_train)

pick = open('model.sav', 'rb')
# pickle.dump(model, pick)
model = pickle.load(pick)
pick.close()
prediction = model.predict(X_test)

accuracy = model.score(X_test, Y_test)

categories = ['Cat', 'Dog']


print('Accuracy = ', accuracy)
print('Prediction is: ', categories[prediction[0]])

mypet = X_test[0].reshape(50, 50)
plt.imshow(mypet, cmap='gray')
plt.show()