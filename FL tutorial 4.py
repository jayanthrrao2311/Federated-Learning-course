#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flwr.client.mod import adaptiveclipping_mod
from flwr.server.strategy import (
    DifferentialPrivacyClientSideAdaptiveClipping,
    FedAvg,
)

from utils4 import *


# In[2]:


def load_data(partition_id):
    fds = FederatedDataset(dataset="mnist", partitioners={"train": 10})
    partition = fds.load_partition(partition_id)

    traintest = partition.train_test_split(test_size=0.2, seed=42)
    traintest = traintest.with_transform(normalize)
    trainset, testset = traintest["train"], traintest["test"]

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64)
    return trainloader, testloader


# In[3]:


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, testloader):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train_model(self.net, self.trainloader)
        return get_weights(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = evaluate_model(self.net, self.testloader)
        return loss, len(self.testloader), {"accuracy": accuracy}


def client_fn(context: Context) -> Client:
    net = SimpleModel()
    partition_id = int(context.node_config["partition-id"])
    trainloader, testloader = load_data(partition_id=partition_id)
    return FlowerClient(net, trainloader, testloader).to_client()


# In[4]:


client = ClientApp(
    client_fn,
    mods=[adaptiveclipping_mod],  # modifiers
)


# In[5]:


net = SimpleModel()
params = ndarrays_to_parameters(get_weights(net))

def server_fn(context: Context):
    fedavg_without_dp = FedAvg(
        fraction_fit=0.6,
        fraction_evaluate=1.0,
        initial_parameters=params,
    )
    fedavg_with_dp = DifferentialPrivacyClientSideAdaptiveClipping(
        fedavg_without_dp,  # <- wrap the FedAvg strategy
        noise_multiplier=0.3,
        num_sampled_clients=6,
    )
    
    # Adjust to 50 rounds to ensure DP guarantees hold
    # with respect to the desired privacy budget
    config = ServerConfig(num_rounds=5)
    
    return ServerAppComponents(
        strategy=fedavg_with_dp,
        config=config,
    )


# In[6]:


server = ServerApp(server_fn=server_fn)


# In[7]:


run_simulation(server_app=server,
               client_app=client,
               num_supernodes=10,
               backend_config=backend_setup
               )


# In[ ]:




