
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential, model_from_json

(xtrain, ytrain),(xtest,ytest)=tf.keras.datasets.mnist.load_data(
    path='mnist.npz'
)
xtrain = xtrain/255
label_list= ["0","1","2","3","4","5","6","7","8","9",]

xtest = xtest/255
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
inp = (xtest)
loaded_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
prediction = loaded_model.predict(inp)
# sample output

print(prediction[35])
print(np.argmax(prediction[35]))
fig = plt.figure()
plt.imshow(xtest[35], cmap='Greys')
plt.title("label:{}".format(label_list[ytest[35]]))
plt.show()