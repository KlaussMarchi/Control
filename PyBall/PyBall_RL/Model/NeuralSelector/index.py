import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicMlpActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hiddenSizes=[400, 300]):
        super(DynamicMlpActor, self).__init__()
        self.layers = nn.ModuleList()
        
        lastDim = state_dim
        for size in hiddenSizes:
            self.layers.append(nn.Linear(lastDim, size))
            lastDim = size
            
        self.outLayer = nn.Linear(lastDim, action_dim)
        self.max_action = max_action
        
    def forward(self, state):
        a = state
        for layer in self.layers:
            a = F.relu(layer(a))
        return self.max_action * torch.tanh(self.outLayer(a))

class DynamicMlpCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hiddenSizes=[400, 300]):
        super(DynamicMlpCritic, self).__init__()
        self.layers = nn.ModuleList()
        
        lastDim = state_dim + action_dim
        for size in hiddenSizes:
            self.layers.append(nn.Linear(lastDim, size))
            lastDim = size
            
        self.outLayer = nn.Linear(lastDim, 1)
        
    def forward(self, state, action):
        q = torch.cat([state, action], 1)
        for layer in self.layers:
            q = F.relu(layer(q))
        return self.outLayer(q)

class DynamicLstmActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hiddenSize=256, numLayers=1):
        super(DynamicLstmActor, self).__init__()
        self.lstm = nn.LSTM(state_dim, hiddenSize, numLayers, batch_first=True)
        self.outLayer = nn.Linear(hiddenSize, action_dim)
        self.max_action = max_action
        
    def forward(self, state):
        # State shape expected to be (batch, seq, state_dim). 
        # If (batch, state_dim) is passed, we unsqueeze it.
        if len(state.shape) == 2:
            state = state.unsqueeze(1)
            
        lstmOut, _ = self.lstm(state)
        # Take the output of the last sequence step
        a = lstmOut[:, -1, :] 
        return self.max_action * torch.tanh(self.outLayer(a))

class DynamicLstmCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hiddenSize=256, numLayers=1):
        super(DynamicLstmCritic, self).__init__()
        self.lstm = nn.LSTM(state_dim + action_dim, hiddenSize, numLayers, batch_first=True)
        self.outLayer = nn.Linear(hiddenSize, 1)
        
    def forward(self, state, action):
        if len(state.shape) == 2:
            state = state.unsqueeze(1)
        if len(action.shape) == 2:
            action = action.unsqueeze(1)
            
        q = torch.cat([state, action], 2)
        lstmOut, _ = self.lstm(q)
        q = lstmOut[:, -1, :]
        return self.outLayer(q)

class NeuralSelector:
    """
    Selector for different Neural Network architectures for DRL.
    """
    options = {
        'mlp_standard': {
            'actorClass':  DynamicMlpActor,
            'criticClass': DynamicMlpCritic,
            'params': {
                'hiddenSizes': [[400, 300], [256, 256], [128, 128]],
                'actorLr': [1e-4, 5e-5],
                'criticLr': [1e-3, 5e-4],
                'batchSize': [64, 128]
            }
        },
        'mlp_deep': {
            'actorClass': DynamicMlpActor,
            'criticClass': DynamicMlpCritic,
            'params': {
                'hiddenSizes': [[400, 300, 200], [256, 256, 128], [512, 256, 128]],
                'actorLr': [1e-4, 1e-5],
                'criticLr': [1e-3, 1e-4],
                'batchSize': [64, 128, 256]
            }
        },
        'mlp_wide': {
            'actorClass': DynamicMlpActor,
            'criticClass': DynamicMlpCritic,
            'params': {
                'hiddenSizes': [[1024, 512], [512, 512]],
                'actorLr': [5e-5, 1e-5],
                'criticLr': [5e-4, 1e-4],
                'batchSize': [128, 256]
            }
        },
        'lstm': {
            'actorClass': DynamicLstmActor,
            'criticClass': DynamicLstmCritic,
            'params': {
                'hiddenSize': [128, 256, 512],
                'numLayers': [1, 2],
                'actorLr': [1e-4, 5e-5],
                'criticLr': [1e-3, 5e-4],
                'batchSize': [64, 128]
            }
        }
    }

    def __init__(self, name: str):
        self.chosen = name
        if name not in self.options:
            raise ValueError(f"Model '{name}' not found. Available options: {list(self.options.keys())}")
        self.selected = self.options[name]

    def get(self):
        """
        Returns the actor class, critic class, and a dictionary of hyperparameters 
        that can be iterated over (similar to a GridSearch parameter grid).
        """
        actorClass  = self.selected['actorClass']
        criticClass = self.selected['criticClass']
        params = self.selected['params']
        return actorClass, criticClass, params
