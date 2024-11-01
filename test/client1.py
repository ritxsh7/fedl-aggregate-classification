import flwr as fl
from utils import model, getData, showDataDist, getMnistDataSet, plotClientData


model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

x_train, y_train, x_test, y_test = getMnistDataSet()

dist = [4000, 10, 4000, 10, 4000, 10, 4000, 10, 4000, 10]

x_train, y_train = getData(dist, x_train, y_train)
showDataDist(y_train)

results_list = []

# Define Flower client
class FlwrClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self, config):
        """Get parameters of the local model."""
        return model.get_weights()

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        # Update local model parameters
        self.model.set_weights(parameters)

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            32,
            3,
            validation_data=(self.x_test, self.y_test),
            verbose=0
        )
        
        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        print("Local Training Metrics on client 1: {}".format(results))
        results_list.append(results)
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32,verbose=0)
        num_examples_test = len(self.x_test)
        print("Evaluation accuracy on Client 1 after weight aggregation : ", accuracy)
        return loss, num_examples_test, {"accuracy": accuracy}

# Start Flower client
client = FlwrClient(model, x_train, y_train, x_test, y_test)
fl.client.start_client(
        server_address="localhost:8080", 
        client=client
)

plotClientData(results_list)