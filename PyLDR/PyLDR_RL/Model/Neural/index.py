import os
import json
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import sys
from tqdm import tqdm


def getModel(folder, id=1):
    path = f'{folder}/model_{id}'
    with open(f'{path}/info.json', 'r', encoding='utf-8') as file:
        data = json.loads(file.read())
    data['model'] = joblib.load(f'{path}/model.pkl')
    return data

class StatesUpdater:
    def __init__(self, size, initial=0.00):
        self.size   = size
        self.buffer = np.array([initial for i in range(size)])

    def update(self, value):
        for i in range(self.size-1, 0, -1):
            self.buffer[i] = self.buffer[i-1]
        self.buffer[0] = value
        return self.buffer

    def get(self, var='x'):
        data = {}
        data[var] = self.buffer[0]
        for i in range(1, self.size):
            data[f'{var}(n-{i})'] = self.buffer[i]
        return {key: float(val) for key, val in data.items()}

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def act(self, state):
        action_mean = self.actor_mean(state)
        action_std  = torch.exp(self.actor_logstd).expand_as(action_mean)
        dist   = Normal(action_mean, action_std)
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_mean = self.actor_mean(state)
        action_std = torch.exp(self.actor_logstd).expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(state)
        
        return (action_logprobs, state_values, dist_entropy)

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam([
            {'params': self.policy.actor_mean.parameters(), 'lr': lr_actor},
            {'params': [self.policy.actor_logstd], 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            action, action_logprob = self.policy_old.act(state)
        
        self.buffer.states.append(state.squeeze(0))
        self.buffer.actions.append(action.squeeze(0))
        self.buffer.logprobs.append(action_logprob.squeeze(0))

        return action.item()

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.stack(self.buffer.states, dim=0).detach()
        old_actions = torch.stack(self.buffer.actions, dim=0).detach()
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Match state_values tensor dimensions with rewards tensor
            state_values = state_values.view(-1)
                
            ratios = torch.exp(torch.clamp(logprobs - old_logprobs.detach(), -20.0, 20.0))

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

class InternalSystem:
    def __init__(self, model, columns, n_states, init_x=0.0, init_u=0.0):
        self.n_states = n_states
        self.sensor   = StatesUpdater(self.n_states, initial=init_x)     
        self.actuator = StatesUpdater(self.n_states, initial=init_u)   
        self.model = model
        self.cols  = columns

    def update(self, actuatorValue):
        self.actuator.update(actuatorValue)
        data = self.actuator.get('u')
        for i in range(1, self.n_states):
            data[f'x(n-{i})'] = self.sensor.buffer[i-1]
        
        df = pd.DataFrame([data], columns=self.cols)
        response = self.model.predict(df)[0]
        response = np.clip(response, 0.0, 1023.0)
        self.sensor.update(response)
        return self.sensor.buffer[0]

class RLEnvironment:
    def __init__(self, system, n_states, sensor_range, actuator_range, setpoint_path, dt, ts):
        self.system = system
        self.n_states = n_states
        self.sensor_range = sensor_range
        self.actuator_range = actuator_range
        self.setpoint_path = setpoint_path
        self.dt = dt
        self.ts = ts
        
        self.current_step = 0
        self.max_steps = len(self.setpoint_path)
        self.ref = 0.0
        self.alpha = np.exp(-4 * (self.dt / self.ts))

        self.sensor = StatesUpdater(n_states, initial=0.0)     
        self.actuator = StatesUpdater(n_states, initial=0.0)
        self.error = StatesUpdater(1, initial=0.0)
        
    def reset(self):
        self.current_step = 0
        init_x = self.sensor_range[0] + (self.sensor_range[1] - self.sensor_range[0]) * 0.5
        init_u = self.actuator_range[0] + (self.actuator_range[1] - self.actuator_range[0]) * 0.5
        
        self.sensor.buffer = np.array([init_x] * self.n_states)
        self.actuator.buffer = np.array([init_u] * self.n_states)
        self.error.buffer = np.array([0.0])
        
        self.system.sensor.buffer = np.array([init_x] * self.system.n_states)
        self.system.actuator.buffer = np.array([init_u] * self.system.n_states)
        
        self.ref = init_x
        return self._get_state()

    def _get_state(self):
        state = []
        s_min, s_max = self.sensor_range
        s_range = s_max - s_min
        a_min, a_max = self.actuator_range
        a_range = a_max - a_min
        
        state.append(self.error.buffer[0] / s_range)
        for i in range(self.n_states):
            state.append((self.sensor.buffer[i] - s_min) / s_range)
        for i in range(self.n_states - 1):
            state.append((self.actuator.buffer[i] - a_min) / a_range)
        state.append((self.ref - s_min) / s_range)
        return np.array(state, dtype=np.float32)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        max_delta = (self.actuator_range[1] - self.actuator_range[0]) * 0.2 
        delta_u = action * max_delta
        
        u = self.actuator.buffer[0] + delta_u
        u = np.clip(u, self.actuator_range[0], self.actuator_range[1])
        self.actuator.update(u)
        
        x = self.system.update(u)
        self.sensor.update(x)
        
        setpoint = self.setpoint_path[self.current_step]
        self.ref = self.alpha * self.ref + (1 - self.alpha) * setpoint
        
        self.error.update(self.ref - x)
        
        norm_error = abs(self.ref - x) / (self.sensor_range[1] - self.sensor_range[0])
        norm_delta_u = abs(delta_u) / (self.actuator_range[1] - self.actuator_range[0])
        
        reward = -norm_error - 0.01 * (norm_delta_u ** 2)
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return self._get_state(), reward, done, {'u': u, 'x': x, 'ref': self.ref, 'setpoint': setpoint}

class RLModelWrapper:
    def __init__(self, options_path='info.json', system_backup_path='Backup/System'):
        with open(options_path, 'r', encoding='utf-8') as f:
            self.options = json.loads(f.read())
            
        self.dt = self.options.get('dt')
        self.ts = self.options.get('ts')
        self.sensor_range = self.options.get('sensor_range')
        self.actuator_range = self.options.get('actuator_range')
        
        self.n_states_ctl = 5
        self.state_dim = 2 * self.n_states_ctl + 1 # Error + Sensor(5) + Actuator(4) + Ref(1) = 11
        self.action_dim = 1
        
        self.agent = PPOAgent(state_dim=self.state_dim, action_dim=self.action_dim, lr_actor=0.001, lr_critic=0.001, gamma=0.99, K_epochs=40, eps_clip=0.2)
        self.system_backup_path = system_backup_path

    def fit(self, X, y, epochs=500):
        sys_data  = getModel(self.system_backup_path, id=1)
        sys_model = sys_data['model']
        sys_variables = sys_data['variables']
        n_states_sys  = len(sys_variables) - 1
        
        x_max = self.sensor_range[1]
        STEP  = int(10 * self.ts / self.dt)
        setpointPath = [x_max*0.9]*STEP + [x_max*0.5]*STEP + [x_max*0.4]*STEP + [x_max*0.6]*STEP
        
        init_x = self.sensor_range[0] + (self.sensor_range[1] - self.sensor_range[0]) * 0.5
        init_u = self.actuator_range[0] + (self.actuator_range[1] - self.actuator_range[0]) * 0.5
        
        system = InternalSystem(sys_model, sys_variables, n_states_sys, init_x=init_x, init_u=init_u)
        env = RLEnvironment(system, self.n_states_ctl, self.sensor_range, self.actuator_range, setpointPath, self.dt, self.ts)
        
        update_timestep = env.max_steps * 2
        index = 0
        
        for epoch in tqdm(range(1, epochs+1), desc="Training RL Agent"):
            state = env.reset()
            
            for t in range(env.max_steps):
                action = self.agent.select_action(state)
                state, reward, done, info = env.step(action)
                
                self.agent.buffer.rewards.append(reward)
                self.agent.buffer.is_terminals.append(done)
                index = (index + 1)
                
                if index % update_timestep == 0:
                    self.agent.update()
        
                if done:
                    break

        return self
        
    def predict(self, X):
        s_min, s_max = self.sensor_range
        s_range      = (s_max - s_min)
        a_min, a_max = self.actuator_range
        a_range   = (a_max - a_min)
        max_delta = (a_range * 0.2)
        
        err_vals = X['error'].values
        x_vals = X['x'].values
        x_hist = [X[f'x(n-{j})'].values for j in range(1, self.n_states_ctl)]
        u_hist = [X[f'u(n-{j})'].values for j in range(1, self.n_states_ctl)]
        
        states = []
        for i in range(len(X)):
            state = []
            state.append(err_vals[i] / s_range)
            state.append((x_vals[i] - s_min) / s_range)
            for j in range(self.n_states_ctl - 1):
                state.append((x_hist[j][i] - s_min) / s_range)
            for j in range(self.n_states_ctl - 1):
                state.append((u_hist[j][i] - a_min) / a_range)
            
            ref_i = err_vals[i] + x_vals[i]
            state.append((ref_i - s_min) / s_range)
            states.append(state)
            
        states_np = np.array(states, dtype=np.float32)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(states_np)
            action_mean = self.agent.policy.actor_mean(state_tensor)
            
        delta_u = action_mean.numpy().flatten() * max_delta
        return delta_u


class NeuralSelector:
    def __init__(self, name='PPO'):
        self.name   = name
        self.chosen = name

    def get(self):
        return RLModelWrapper(options_path='info.json', system_backup_path='Backup/System')
 