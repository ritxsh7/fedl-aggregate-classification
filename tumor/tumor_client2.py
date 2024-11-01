import os
import flwr as fl
import pandas as pd
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from helpers import create_model, plotMap, plotClientData, load_train_data,load_test_data

# Create data generators
datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load train data
client2_train = load_train_data()
client2_test = load_test_data()

count = client2_train['labels'].value_counts()

plotMap(count)

client2_train_gen = datagen.flow_from_dataframe(
    client2_train,
    x_col = "filepaths",
    y_col = "labels",
    target_size = (244,244),
    color_mode = "rgb",
    class_mode="categorical",
    batch_size = 32
)

client2_test_gen = datagen.flow_from_dataframe(
    client2_test,
    x_col = "filepaths",
    y_col = "labels",
    target_size = (244,244),
    color_mode = "rgb",
    class_mode="categorical",
    batch_size = 32
)


results_list = []

class FlwrClient(fl.client.NumPyClient):
    def __init__(self, model, train_gen, test_gen):
        self.model = model
        self.train_gen = train_gen
        self.test_gen = test_gen

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        history = self.model.fit(self.train_gen, epochs = 1)
        results = {"loss": history.history["loss"][0], "accuracy": history.history["accuracy"][0]}
        print("Local Training Metrics on client 1: {}".format(results))
        results_list.append(results)
        return self.model.get_weights(), len(self.train_gen),results

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.test_gen)
        num_examples_test = len(self.test_gen)
        return loss, num_examples_test, {"accuracy": accuracy}
    

model = create_model()
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

client = FlwrClient(model, client2_train_gen, client2_test_gen)
fl.client.start_client(server_address="localhost:8080", client=client)


plotClientData(results_list)