import flwr as fl
from typing import Dict, Optional, Tuple
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from helpers import plotServerData, create_model

# Create a test generator for evaluation
def create_test_generator(test_dir):
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    return datagen.flow_from_directory(
        test_dir,
        target_size=(244, 244),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )


results_list = []

# Define evaluation function for server-side evaluation
def get_eval_fn(model):
    test_dir = "../data/Testing"
    test_gen = create_test_generator(test_dir)

    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters) 
        loss, accuracy = model.evaluate(test_gen)
        print("After round {}, Global accuracy = {} ".format(server_round,accuracy))
        results = {"round":server_round,"loss": loss, "accuracy": accuracy}
        results_list.append(results)
        return loss, {"accuracy": accuracy}

    return evaluate

# Initialize model and compile
model = create_model()
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

# Define strategy
strategy = fl.server.strategy.FedAvg(evaluate_fn=get_eval_fn(model), min_available_clients = 3)

# Start Flower server
fl.server.start_server(

    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy
)

model.save("../model/final_server_model.h5") 

plotServerData(results_list)
