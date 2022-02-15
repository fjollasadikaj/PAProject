import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout


data = []
labels = []
classes = 43
cur_path = os.getcwd()

for i in range(classes):
    path = os.path.join(cur_path, 'train', str(i))
    images = os.listdir(path)

    for a in images:
        try:
            images = Image.open(path + '\\'+ a)
            image = image.resize((30,30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print("Gabim gjatë marrjes së imazhit")

# Konvertimi i listave nepermjet numpy ne vargje
data = np.array(data)
labels = np.array(labels)

print(data.shape, labels.shape)
#Ndarja e të dhënave të trajnimit dhe testimi
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

#Ndertimi i modelit sekuencial
model = Sequential()
model.add(Conv2D(filter=32, kernel_size(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filter=32, kernel_size(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs = 15
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
model.save("traffic_classifier.h5")


#Vizualizimi i grafikoneve per saktesine e modelit bazuar ne te dhenat per trajnim dhe testim

plt.figure(0)
plt.plot(history['accurancy'], label='training accurancy')
plt.plot(history['val_accurancy'], label='val_accurancy')
plt.title('Accurancy-Saktësia')
plt.xlabel('epochs')
plt.ylabel('accurancy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history['loss'], label='training loss')
plt.plot(history['val_loss'], label='val_loss')
plt.title('Loss-Humbjet')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

#Testimi i saktesise ne dataset
from sklearn.metrics import accurancy_score
y_test = pd.read_csv('Test.csv')
labels = y_test["ClassIs"].values
imgs = y_test["Path"].values

data =[]
for img in imgs:
    image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))

X_test = np.array(data)


pred = model.predict_classes(X_test)

from sklearn.metrics import accurancy_score
print(accurancy_score(labels, pred))

