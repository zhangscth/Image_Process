#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_acc_loss():
    f = open("./plot/adam_metric.txt", 'rb')
    ADAM = pickle.load(f)
    f = open("./plot/sgd_metric.txt", 'rb')
    SGD = pickle.load(f)
    f = open("./plot/adagrad_metric.txt", 'rb')
    Adagrad = pickle.load(f)
    f = open("./plot/rmsprop_metric.txt", 'rb')
    RMSProp = pickle.load(f)

    f.close()
    fig = plt.figure(2)
    #loss
    optimizer=["ADAM","SGD","Adagrad","RMSProp"]
    for i,op in enumerate([ADAM,SGD,Adagrad,RMSProp]):
        x= range(100)

        fig.add_subplot(2,4,i+1)
        plot1, = plt.plot(x, op["train_loss"])
        plot2, = plt.plot(x, op["test_loss"])
        plt.title(optimizer[i])
        plt.ylim(0.4, 1)
        plt.legend([plot1, plot2], ["train loss", "test loss"])

    for i,op in enumerate([ADAM,SGD,Adagrad,RMSProp]):
        x= range(100)

        fig.add_subplot(2,4,4+i+1)
        plot1, = plt.plot(x, op["train_acc"])
        plot2, = plt.plot(x, op["test_acc"])
        plt.title(optimizer[i])
        plt.ylim(0.4,1)
        plt.legend([plot1, plot2], ["train acc", "test acc"])
    plt.show()


def batch_plot():
    f = open("./batch_size/adam32_metric.txt", 'rb')
    ADAM = pickle.load(f)
    f = open("./batch_size/adam64_metric.txt", 'rb')
    SGD = pickle.load(f)
    f = open("./batch_size/adam96_metric.txt", 'rb')
    Adagrad = pickle.load(f)
    f = open("./batch_size/adam128_metric.txt", 'rb')
    RMSProp = pickle.load(f)

    f.close()
    fig = plt.figure(2)
    # loss
    optimizer = ["batch_size:32", "batch_size:64", "batch_size:96", "batch_size:128"]
    for i, op in enumerate([ADAM, SGD, Adagrad, RMSProp]):
        x = range(100)

        fig.add_subplot(2, 4, i + 1)
        plot1, = plt.plot(x, op["train_loss"])
        plot2, = plt.plot(x, op["test_loss"])
        plt.title(optimizer[i])
        plt.ylim(0.4, 1)
        plt.legend([plot1, plot2], ["train loss", "test loss"])

    for i, op in enumerate([ADAM, SGD, Adagrad, RMSProp]):
        x = range(100)

        fig.add_subplot(2, 4, 4 + i + 1)
        plot1, = plt.plot(x, op["train_acc"])
        plot2, = plt.plot(x, op["test_acc"])
        plt.title(optimizer[i])
        plt.ylim(0.4, 1)
        plt.legend([plot1, plot2], ["train acc", "test acc"])
    plt.show()

if __name__=="__main__":
    plot_acc_loss()
    batch_plot()
