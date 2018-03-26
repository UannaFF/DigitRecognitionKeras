import numpy
import sys
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

if len(sys.argv) != 4:
    print("Usage: file.py numNeuron epochsNumb batchSize")
    sys.exit()

seed = 7
numpy.random.seed(seed)

numNeuron = int(sys.argv[1])
numClass = 10
pixels = 784

#define model
def model():
    model = Sequential()
    model.add(Dense(numNeuron, input_dim=pixels, activation="relu"))
    model.add(Dropout(0.2, noise_shape=None, seed=seed))
    model.add(Dense(numClass, activation="softmax"))

    #model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

model = model()
epochsNumb = int(sys.argv[2])
batchSize = int(sys.argv[3])
# Fit the model
model.fit(X_train, y_train, epochs=epochsNumb, batch_size=batchSize)

# evaluate the model
scores = model.evaluate(X_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("\nEpoch: %d, Batch size: %d" % (epochsNumb, batchSize))

print("\nMaking predictions\n")
# calculate predictions
predictions = model.predict(X_test)
# Compare predictions
class_counter = [0] * 10
for x,y in zip(predictions,y_test):
    # print wrong predictions
    for i in range(10):
        if round(x[i]) == 1 and round(x[i]) != round(y[i]):
            class_counter[i] = class_counter[i] + 1
            # print("Failed")
            # print("[", end="")
            # for i in range(10):
            #     if i != 9:
            #         print("%d-" % round(x[i]), end="")
            #     else:
            #         print("%d]" % round(x[i]))
            # # print y_test
            # print("[", end="")
            # for i in range(10):
            #     if i != 9:
            #         print("%d-" % round(y[i]), end="")
            #     else:
            #         print("%d]" % round(y[i]))
            # print("\n")
            break

print("Class error counter: ", end="")
print(class_counter)
