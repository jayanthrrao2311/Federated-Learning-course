#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset

from utils3 import *


# In[2]:


def load_data(partition_id):
    fds = FederatedDataset(dataset="mnist", partitioners={"train": 5})
    partition = fds.load_partition(partition_id)

    traintest = partition.train_test_split(test_size=0.2, seed=42)
    traintest = traintest.with_transform(normalize)
    trainset, testset = traintest["train"], traintest["test"]

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64)
    return trainloader, testloader


# In[3]:


def fit_config(server_round: int):
    config_dict = {
        "local_epochs": 2 if server_round < 3 else 5,
    }
    return config_dict


# In[4]:


net = SimpleModel()
params = ndarrays_to_parameters(get_weights(net))

def server_fn(context: Context):
    strategy = FedAvg(
        min_fit_clients=5,
        fraction_evaluate=0.0,
        initial_parameters=params,
        on_fit_config_fn=fit_config,  # <- NEW
    )
    config=ServerConfig(num_rounds=3)
    return ServerAppComponents(
        strategy=strategy,
        config=config,
    )


# In[5]:


server = ServerApp(server_fn=server_fn)


# In[6]:


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, testloader):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader

    def fit(self, parameters, config):
        set_weights(self.net, parameters)

        epochs = config["local_epochs"]
        log(INFO, f"client trains for {epochs} epochs")
        train_model(self.net, self.trainloader, epochs)

        return get_weights(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = evaluate_model(self.net, self.testloader)
        return loss, len(self.testloader), {"accuracy": accuracy}


# In[7]:


def client_fn(context: Context) -> Client:
    net = SimpleModel()
    partition_id = int(context.node_config["partition-id"])
    trainloader, testloader = load_data(partition_id=partition_id)
    return FlowerClient(net, trainloader, testloader).to_client()


client = ClientApp(client_fn)


# In[8]:


run_simulation(server_app=server,
               client_app=client,
               num_supernodes=5,
               backend_config=backend_setup
               )


# In[ ]:




