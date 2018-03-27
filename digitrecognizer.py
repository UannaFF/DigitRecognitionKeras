import numpy as np
import sys
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import keras.callbacks
from keras.callbacks import TensorBoard
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from keras import backend as K
K.set_image_dim_ordering('th')


def model(num_neurons, num_classes, num_pixels, shape):
    model = Sequential()
    model.add(Dense(num_neurons, input_dim=num_pixels, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def runDataAugmentation(use_model, X, Y, epochs, batchsize, x_test, y_test):
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    X = reshapeFourDimensional(X)
    datagen.fit(X)
    
    for e in range(epochs):
        print('Epoch', e)
        batches = 0
        for x_batch, y_batch in datagen.flow(X, Y, batch_size=batchsize):
            x_batch = reshapeOneDimensional(x_batch)
            y_batch = np_utils.to_categorical(y_batch)
            use_model.fit(x_batch, y_batch)
            batches += 1
            # evaluate the model
            stats = use_model.evaluate(x_batch, y_batch)
            print("stats for batch "+str(batches)+": "+str(stats))
            if batches >= len(x_batch) / batchsize:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break

def getFailCases(prediction, validation):
    errors_counter = [0] * 10
    count = 0;
    for p, v in zip(prediction, validation):
        #If we compare and it's different, it's a fail case
        if(np.argmax(p) != np.argmax(v)):
            print("Good: "+str(np.argmax(v))+" bad: "+str(np.argmax(p)))
            errors_counter[np.argmax(v)]+=1 #count the error on the value that should've been
            count += 1
    print(str(errors_counter))
    print("Length of tests: "+str(len(validation))+"total erros: "+str(count) + " percentage: "+str(count/len(validation)))




def reshapeFourDimensional(data):
    return data.reshape(data.shape[0], 1,data.shape[1], data.shape[2])

#flatten 28*28 images to a 784 vector for each image
def reshapeOneDimensional(data):
    num_data = data.shape[2] * data.shape[2]
    return data.reshape(data.shape[0], num_data).astype('float32')

def plotHistory(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def showFirstElements(X, Y):
    print(Y[0])
    print(Y[1])
    print(Y[2])
    print(Y[3])
    # plot 4 images as gray scale
    #plt.subplot(221)
    #plt.imshow(X[0], cmap=plt.get_cmap('gray'))
    #plt.subplot(222)
    #plt.imshow(X[1], cmap=plt.get_cmap('gray'))
    #plt.subplot(223)
    #plt.imshow(X[2], cmap=plt.get_cmap('gray'))
    #plt.subplot(224)
    #plt.imshow(X[3], cmap=plt.get_cmap('gray'))
    # show the plot
    #plt.show()

def main():

    if len(sys.argv) != 4:
        print("Usage: file.py numNeuron epochsNumb batchSize")
        sys.exit()

    num_neuron = int(sys.argv[1])
    num_epochs = int(sys.argv[2])
    batch_size = int(sys.argv[3])

    # load data
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    orig_train_x = X_train
    orig_train_y = Y_train

    showFirstElements(X_test, Y_test)
    
    #reshape
    X_train = reshapeOneDimensional(X_train)
    X_test = reshapeOneDimensional(X_test)

    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255

    # one hot encode outputs
    Y_train = np_utils.to_categorical(Y_train)
    Y_test = np_utils.to_categorical(Y_test)

    num_classes = Y_test.shape[1]

    use_model = model(num_neuron, num_classes, X_train.shape[1], X_train.shape)

    use_model.summary() #print summary of model

    use_model.save_weights("./weights.txt")
    #use_model.load_weights(filepath, by_name=False)

    # Fit the model
    history = use_model.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size, 
        shuffle=False)

    #plotHistory(history)

    # evaluate the model
    stats = use_model.evaluate(X_train, Y_train)

    print("stats first model: "+str(stats))

    # calculate predictions
    predictions = use_model.predict_classes(X_test, verbose=1)
    #print(str(predictions))
    #getFailCases(predictions, classes)

    #Reload initial weights to test with augmentation
    use_model.load_weights("./weights.txt", by_name=False)

    #Augment and train
    runDataAugmentation(use_model, orig_train_x, 
        orig_train_y, num_epochs, batch_size, X_test,Y_test)


main()