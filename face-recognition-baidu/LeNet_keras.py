import keras
import time
from load_data import load_data
from keras import optimizers
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard


def build_model():
    model = Sequential()
    model.add(Conv2D(6, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', input_shape=(32,32,3)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(16, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation = 'relu', kernel_initializer='he_normal'))
    model.add(Dense(84, activation = 'relu', kernel_initializer='he_normal'))
    model.add(Dense(6, activation = 'softmax', kernel_initializer='he_normal'))
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def scheduler(epoch):
    if epoch < 100:
        return 0.01
    if epoch < 150:
        return 0.005
    return 0.001


if __name__ == '__main__':
    t0 = time.time()
    # load data 
    # x are image datas, y are the labels
    dataset_path = "dataset"
    (x_train, y_train), (x_test, y_test) = load_data(dataset_path)
    y_train = keras.utils.to_categorical(y_train, 6)
    y_test = keras.utils.to_categorical(y_test, 6)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0

    # build network
    model = build_model()
    print(model.summary())

    # set callback
    # 查看训练的损失函数曲线命令：tensorboard --logdir="C:\Users\iweut\PythonTest\deeplearning\face-recognition-baidu\lenet"
    tb_cb = TensorBoard(log_dir='lenet', histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr,tb_cb]

    # start train
    model.fit(x_train, y_train,
              batch_size=32,
              epochs=20,
              callbacks=cbks,
              validation_data=(x_test, y_test),
              shuffle=True)

    # save model
    model.save('lenet.h5')
    print("The excution time of program is {} seconds.".format(time.time()-t0))