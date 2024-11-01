import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential


def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(244, 244, 3)),  # Adjust input shape here
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(4, activation="softmax")  # 4 classes
    ])
    return model


def plotMap(data):
    plt.figure(figsize=(10, 6))
    data.plot(kind='bar', color='skyblue')
    plt.title('Distribution of Tumor Types in Training Data')
    plt.xlabel('Tumor Type')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plotServerData(data):
    df = pd.DataFrame(data)
    plt.plot(df['loss'],color = 'b', label = 'loss')
    plt.plot(df['accuracy'],color = 'g', label = 'accuracy')
    plt.legend(loc = 'lower right')
    plt.xlabel('Rounds')
    plt.ylabel('accuracy')
    plt.show()


def plotClientData(data):
    df = pd.DataFrame(data)
    plt.plot(df['accuracy'],color = 'b', label = 'training accuracy')
    # plt.plot(df['val_accuracy'],color = 'g', label = 'validation accuracy')
    plt.legend(loc = 'lower right')
    plt.xlabel('Rounds')
    plt.ylabel('accuracy')
    plt.show()


train_dir = "../data/Training"
test_dir = "../data/Testing"


def load_data(dir):
    filepaths = []
    labels = []


    folds = os.listdir(dir)

    for fold in folds:
        foldpath = os.path.join(dir, fold)
        
        files = os.listdir(foldpath)
        for f in files:
            fpath = os.path.join(foldpath, f)
            
            filepaths.append(fpath)
            labels.append(fold)

    return pd.DataFrame(data={'filepaths':filepaths, 'labels':labels})


def load_train_data():
    client_data = load_data(train_dir)
    return client_data.sample(frac=0.5, random_state=np.random.randint(1, 100))
    

def load_test_data():
    return load_data(test_dir)