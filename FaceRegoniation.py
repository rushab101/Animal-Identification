import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle



DATADIR= "E:/Python Programs/ImageRecgonization/Cats and Dogs"
CATEGORIES = ["Dog", "Cat", "monkey","squirrel"]
for category in CATEGORIES:  # do dogs and cats
    path = os.path.join(DATADIR,category)  # create path to dogs and cats
    for img in os.listdir(path):  # iterate over each image per dogs and cats
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array

IMG_SIZE = 50

#new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

training_data = []
def create_training_data():
    for category in CATEGORIES:  # do dogs, cats, monkeys, squirrel

        path = os.path.join(DATADIR,category)  # create path to dogs, cats, monkeys, squirrel
        class_num = CATEGORIES.index(category)  # get the classification  (0,1,2,3). 0=dog 1=cat 2=monkey 3=squirrel

        for img in os.listdir(path):
            try:
              img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
              new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
              training_data.append([new_array, class_num])
            except Exception as e:  # in the interest in keeping the output clean...
                pass
create_training_data()
print(len(training_data))
random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])

X= []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

pickle_out = open("X.pickle_new","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle_new","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

