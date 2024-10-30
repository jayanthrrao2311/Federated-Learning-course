#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flwr.client.mod import parameters_size_mod

from utils5 import *


# In[2]:


model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-14m",
    cache_dir="./pythia-14m/cache",
)


# In[3]:


vals = model.state_dict().values()
total_size_bytes = sum(p.element_size() * p.numel() for p in vals)
total_size_mb = int(total_size_bytes / (1024**2))

log(INFO, "Model size is: {} MB".format(total_size_mb))


# In[4]:


class FlowerClient(NumPyClient):
    def __init__(self, net):
        self.net = net

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        # No actual training here
        return get_weights(self.net), 1, {}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        # No actual evaluation here
        return float(0), int(1), {"accuracy": 0}


def client_fn(context: Context) -> FlowerClient:
    return FlowerClient(model).to_client()


client = ClientApp(
    client_fn,
    mods=[parameters_size_mod],
)


# In[5]:


bandwidth_sizes = []


class BandwidthTrackingFedAvg(FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Track sizes of models received
        for _, res in results:
            ndas = parameters_to_ndarrays(res.parameters)
            size = int(sum(n.nbytes for n in ndas) / (1024**2))
            log(INFO, f"Server receiving model size: {size} MB")
            bandwidth_sizes.append(size)

        # Call FedAvg for actual aggregation
        return super().aggregate_fit(server_round, results, failures)

    def configure_fit(self, server_round, parameters, client_manager):
        # Call FedAvg for actual configuration
        instructions = super().configure_fit(
            server_round, parameters, client_manager
        )

        # Track sizes of models to be sent
        for _, ins in instructions:
            ndas = parameters_to_ndarrays(ins.parameters)
            size = int(sum(n.nbytes for n in ndas) / (1024**2))
            log(INFO, f"Server sending model size: {size} MB")
            bandwidth_sizes.append(size)

        return instructions


# In[6]:


params = ndarrays_to_parameters(get_weights(model))

def server_fn(context: Context):
    strategy = BandwidthTrackingFedAvg(
        fraction_evaluate=0.0,
        initial_parameters=params,
    )
    config = ServerConfig(num_rounds=1)
    return ServerAppComponents(
        strategy=strategy,
        config=config,
    )


server = ServerApp(server_fn=server_fn)


# In[7]:


run_simulation(server_app=server,
               client_app=client,
               num_supernodes=2,
               backend_config=backend_setup
               )


# In[8]:


log(INFO, "Total bandwidth used: {} MB".format(sum(bandwidth_sizes)))


# In[ ]:




