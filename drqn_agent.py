from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.utils.adx.structures import Bid, BidBundle, MarketSegment 

import random
import numpy as np

import os

from collections import deque

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.optim import Adam
import torch.nn.functional as F

'''
IMPORTANT: DATA DEFINITIONS
  1. This model will train to generate q-values for a ad-bidding agent
  2. The input will be given as a sequence of vectors, each of which is a state
  3. The state is composed of the following
      [campaign progress, effective reach, campaign reach, campaign budget, accumulated cost, demographic, time_remaining]
'''

class RNNModel(nn.Module):
    def __init__(self, hidden_size=512, input_vec_len=7):
        super().__init__()
    
        self.rnn = nn.LSTM(input_size=input_vec_len, hidden_size=hidden_size, num_layers=RNN_NUM_LAYERS, batch_first=True)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,NUM_POSSIBLE_ACTIONS),
        )
    
    def forward(self, x):
        _, (h_n, _) = self.rnn(x)

        last_hidden = h_n[-1]

        return self.mlp(last_hidden)
        

class DRQNAgent(NDaysNCampaignsAgent):
    def __init__(self, name):
        super().__init__()

        self.name = name
        
        self.training_mode = True
        self.training_cycles = 100

        self.impression_rnn_model = RNNModel()
        self.impression_model_optimizer = Adam(self.impression_rnn_model.parameters(), lr=LEARNING_RATE)
        self.active_impression_episodes = {}
        self.impression_memory = deque()

        self.prev_profit = 0


    '''
    Method to set bids on impression opportunities; includes drqn
    '''
    def get_ad_bids(self):
        # first thing to do is update past values
        if self.get_current_day() != 1:
            self.add_reward_to_impression_experiences()
            self.archive_impression_episodes()
            self.prev_profit = self.get_cumulative_profit()

            if self.training_mode:
                self.training_step(self.training_cycles)
        
        # then we add new states
        self.add_impression_experience()
        
        bundles = set()
        
        # model will decide what percentage of budget to spend on ALL valid subsets for a target_segment. 
        # This action is added to the experience here.
        for campaign in self.get_active_campaigns():
            bid_entries = set()
        
            sequence = self.active_impression_episodes[campaign.uid]
            state_seq = []
            for s in sequence:
                state_seq.append(s[0])
            
            with torch.no_grad():
                q_values = self.impression_rnn_model(torch.tensor(state_seq).unsqueeze(0))
            
            bid_idx = np.argmax(q_values).item()
            if random.random() <= EPSILON:
                bid_idx = random.randint(1,NUM_POSSIBLE_ACTIONS)

            self.active_impression_episodes[campaign.uid][0][1] = bid_idx
                    
            bid_value = ((bid_idx + 1) / NUM_POSSIBLE_ACTIONS) * campaign.budget
            
            for segment in MarketSegment.all_segments():
                if campaign.target_segment.issubset(segment) or segment == campaign.target_segment:
                    limit = campaign.budget / (campaign.end_day - campaign.start_day + 1)
                    bid = Bid(self, segment, bid_value, limit)
                else:
                    bid = Bid(self, segment, 0, 0)
                
                bid_entries.add(bid)

            bundle = BidBundle(campaign.uid, campaign.budget, bid_entries)
            bundles.add(bundle)

        return bundles
    
    
    def training_step(self, num_cycles):
        for _ in range(num_cycles):
            batch_size = min(BATCH_SIZE, len(self.impression_memory))
            
            if batch_size == 0:
                break
            
            minibatch = []
            
            for _ in range(batch_size):
                i = random.randint(0,len(self.impression_memory)-1)
                minibatch.append(self.impression_memory[i])
            
            state_sequences = []
            q_values_obs = []
            actions = []
            
            for episode in minibatch:
                end_idx = 0
                model_input = []
                ep_state_seq = []

                # grow window to add feautre data. 
                # feature with 1 experience, then 2, then 3, etc. until campaign is over
                while end_idx < len(episode):
                    ep_state_seq.append(episode[end_idx][0])
                    state_sequences.append(ep_state_seq)
                    
                    
                    if end_idx == len(episode) - 1:
                        target = episode[end_idx][2]
                    else:
                        model_input.append(episode[end_idx][3])
                        
                        pred_q_val = self.impression_rnn_model(torch.tensor(model_input).unsqueeze(0))
                        pred_q_val = max(pred_q_val.squeeze(0)).item()
                        
                        target = episode[end_idx][2] + GAMMA * pred_q_val
                    
                    
                    if episode[end_idx][1]:
                        q_values_obs.append(target)
                        actions.append(episode[end_idx][1])
                    
                    end_idx += 1
            
            actions = torch.tensor(actions).unsqueeze(1)
            
            lengths = []
            for seq in state_sequences:
                lengths.append(len(seq)) 
            
            
            lengths = torch.tensor(lengths)
            
            tensor_state_sequences = []
            for seq in state_sequences:
                tensor_state_sequences.append(torch.tensor(seq)) 
            
            padded_state_sequences = pad_sequence(tensor_state_sequences, batch_first=True)
            packed_state_sequences = pack_padded_sequence(padded_state_sequences, lengths, batch_first=True, enforce_sorted=False)
            q_values_pred = self.impression_rnn_model(packed_state_sequences).gather(1, actions).squeeze(1)

            q_values_obs = torch.tensor(q_values_obs)

            mse = ((q_values_pred - q_values_obs)**2).mean()

            self.impression_model_optimizer.zero_grad()
            mse.backward()
            self.impression_model_optimizer.step()
   

    def hash_target_segment(self, segment):
        mappings = {
            MarketSegment(("Male", "Young")): 0,
            MarketSegment(("Male", "Old")): 1,
            MarketSegment(("Male", "LowIncome")): 2,
            MarketSegment(("Male", "HighIncome")): 3,
            MarketSegment(("Female", "Young")): 4,
            MarketSegment(("Female", "Old")): 5,
            MarketSegment(("Female", "LowIncome")): 6,
            MarketSegment(("Female", "HighIncome")): 7,
            MarketSegment(("Young", "LowIncome")): 8,
            MarketSegment(("Young", "HighIncome")): 9,
            MarketSegment(("Old", "LowIncome")): 10,
            MarketSegment(("Old", "HighIncome")): 11,
            MarketSegment(("Male", "Young", "LowIncome")): 12,
            MarketSegment(("Male", "Young", "HighIncome")): 13,
            MarketSegment(("Male", "Old", "LowIncome")): 14,
            MarketSegment(("Male", "Old", "HighIncome")): 15,
            MarketSegment(("Female", "Young", "LowIncome")): 16,
            MarketSegment(("Female", "Young", "HighIncome")): 17,
            MarketSegment(("Female", "Old", "LowIncome")): 18,
            MarketSegment(("Female", "Old", "HighIncome")): 19
        }
        
        return mappings[segment]
    

    '''
    An episode is a set of experiences, defined as:
        (current state, action, reward, new state)
    An impression state is defined as:
        (campaign progress, effective reach, campaign reach, campaign budget, accumulated cost, demographic, time_remaining)
    '''
    def add_impression_experience(self):
        campaigns = self.get_active_campaigns()
        for campaign in campaigns:
            current_real_reach = self.get_cumulative_reach(campaign)
            current_effective_reach = self.effective_reach(current_real_reach, campaign.reach)
            current_cost = self.get_cumulative_cost(campaign)
            target_segment = self.hash_target_segment(campaign.target_segment)
            time_remaining = campaign.end_day - campaign.start_day + 1
            
            state = [current_real_reach, current_effective_reach, campaign.reach, campaign.budget, current_cost, target_segment, time_remaining]
            
            new_experience = [state,None,None,None] # state, action, reward, new_state
            
            if campaign.uid not in self.active_impression_episodes:
                self.active_impression_episodes[campaign.uid] = [new_experience]
            else:
                self.active_impression_episodes[campaign.uid].append(new_experience)
            
            if len(self.active_impression_episodes[campaign.uid]) > 1:
                self.active_impression_episodes[campaign.uid][-2][3] = state
    
    
    '''
    Function to calculate the profit and update the current episodes
    '''
    def add_reward_to_impression_experiences(self):
        active_campaigns = {}
        for campaign in self.get_active_campaigns():
            active_campaigns[campaign.uid] = campaign
        
        profit = (self.get_cumulative_profit() - self.prev_profit) * self.get_quality_score()

        for impression in list(self.active_impression_episodes.keys()):
            # change reward based on if campaign is over
            if impression in active_campaigns:
                # if not, reward is change in completion scaled by budget remaining
                campaign = active_campaigns[impression]
                
                prev_effective_reach = self.active_impression_episodes[impression][-1][0][1]
                current_effective_reach = self.effective_reach(self.get_cumulative_reach(campaign), campaign.reach)
                reward = (current_effective_reach - prev_effective_reach) * (campaign.budget - self.get_cumulative_cost(campaign))

                self.active_impression_episodes[impression][-1][2] = reward
            else:
                # if it is, reward is profit
                self.active_impression_episodes[impression][-1][2] = profit


    '''
    Archives inactive campaigns, and removes old ones if exceeding memory size
    '''
    def archive_impression_episodes(self):
        uids = set()
        for campaign in self.get_active_campaigns():
            uids.add(campaign.uid)
        
        for episode in list(self.active_impression_episodes.keys()):
            if episode not in uids:
                self.impression_memory.append(self.active_impression_episodes[episode])
                
                if len(self.impression_memory) > IMPRESSION_MEMORY_SIZE:
                    self.impression_memory.popleft()
                
                self.active_impression_episodes.pop(episode)



'''
HYPERPARAMETERS 
'''

# RNN HYPERPARAMS
RNN_NUM_LAYERS = 2
NUM_POSSIBLE_ACTIONS = 50 # different budget subdivisions to bet

# AGENT HYPERPARAMS
LEARNING_RATE = 0.001
GAMMA = 0.01
EPSILON = 0.05
BATCH_SIZE = 64
IMPRESSION_MEMORY_SIZE = 1000 # 1 unit is 1 epsiode
