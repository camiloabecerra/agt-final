from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.utils.adx.structures import Bid, BidBundle, MarketSegment 
from path_utils import path_from_local_root

import random
import numpy as np

import os

from collections import deque

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.optim import Adam


'''
IMPORTANT: DATA DEFINITIONS
  1. This model will train to generate q-values for a campaign-bidding agent
  2. The input will be given as a sequence of vectors, each of which is a state
  3. The state is composed of the following
      [campaign demographic, campaign reach, campaign length, current day]
  4. Reward will be winning the campaign * budget: We want to win campaigns and have a lot of money to spend on them
'''

class NNModel(nn.Module):
    def __init__(self, input_vec_len=4):
        super().__init__()
    
        self.mlp = nn.Sequential(
            nn.Linear(input_vec_len, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,NUM_POSSIBLE_ACTIONS),
        )
    
    def forward(self, x):
        return self.mlp(x)
        

class DRQNCampaignAgent(NDaysNCampaignsAgent):
    def __init__(self, name):
        super().__init__()

        self.name = name

        self.training_mode = IS_TRAINING
        self.loading_model = LOAD_MODEL
        self.training_cycles = NUM_TRAINING_CYCLES
        
        self.campaign_nn_model = NNModel()
        
        if self.loading_model:
            path = MODEL_SAVE_PATH
            self.campaign_nn_model.load_state_dict(torch.load(path))
        
        self.campaign_model_optimizer = Adam(self.campaign_nn_model.parameters(), lr=LEARNING_RATE)
        self.active_campaign_episodes = {}
        self.memory = deque()

        self.current_epsilon = EPSILON_START


    '''
    Method to set bids on campaigns; includes drqn
    '''
    def get_campaign_bids(self, campaigns_for_auction):
        # first thing to do is update past values
        if self.get_current_day() != 1:
            self.check_yesterday_rewards()
            self.archive_yesterday_episodes()

            if self.training_mode:
                self.training_step(self.training_cycles)
        
        # then we add new states
        self.add_today_episodes(campaigns_for_auction)
        
        bids = {} # campaign; bid

        for campaign in campaigns_for_auction:
            state = self.active_campaign_episodes[campaign.uid][0]
            
            with torch.no_grad():
                q_values = self.campaign_nn_model(torch.tensor(state).unsqueeze(0))
            
            bid_idx = np.argmax(q_values).item()
            
            if not self.training_mode:
                self.current_epsilon = EPSILON_END

            if random.random() <= self.current_epsilon:
                bid_idx = random.randint(0,NUM_POSSIBLE_ACTIONS-1)
            
            if self.current_epsilon > EPSILON_END:
                self.current_epsilon *= EPSILON_DECAY 
            
            self.active_campaign_episodes[campaign.uid][1] = bid_idx
                    
            bid_value = max(((bid_idx + 1) / NUM_POSSIBLE_ACTIONS), 0.1) * campaign.reach 
            
            bids[campaign] = bid_value
        
        return bids
    
    
    def training_step(self, num_cycles):
        for _ in range(num_cycles):
            batch_size = min(BATCH_SIZE, len(self.memory))
            
            if batch_size == 0:
                break
            
            minibatch = []
            
            for _ in range(batch_size):
                i = random.randint(0,len(self.memory)-1)
                minibatch.append(self.memory[i])
            
            states = []
            q_values_obs = []
            actions = []
            
            for episode in minibatch:
                states.append(episode[0])
                actions.append(episode[1])
                q_values_obs.append(episode[2])
            
            actions = torch.tensor(actions).unsqueeze(1)
            states = torch.tensor(states)
            
            q_values_pred = self.campaign_nn_model(states).gather(1, actions).squeeze(1)

            q_values_obs = torch.tensor(q_values_obs)

            mse = ((q_values_pred - q_values_obs)**2).mean()

            self.campaign_model_optimizer.zero_grad()
            mse.backward()
            self.campaign_model_optimizer.step()

            torch.save(self.campaign_nn_model.state_dict(), MODEL_SAVE_PATH)
   

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
    An episode is 1 experience, defined as:
        (current state, action, reward, DONE)
    A state is defined as:
        (campaign demographic, campaign reach, campaign length, current day)
    '''
    def add_today_episodes(self, campaigns_for_auction):
        for campaign in campaigns_for_auction:
            segment = self.hash_target_segment(campaign.target_segment)
            length = campaign.end_day - campaign.start_day + 1
            
            state = [float(segment), float(campaign.reach), float(length), float(self.get_current_day())]
            new_experience = [state,None,None,None] # state, action, reward, new_state
            
            self.active_campaign_episodes[campaign.uid] = new_experience
            
    
    '''
    Function to calculate reward on yesterday's actions
    '''
    def check_yesterday_rewards(self):
        active_campaigns = {}
        for campaign in self.get_active_campaigns():
            active_campaigns[campaign.uid] = campaign
        
        lost_campaigns = {}
        for uid in self.active_campaign_episodes.keys():
            if uid not in active_campaigns:
                lost_campaigns[uid] = self.active_campaign_episodes[uid]
        
        for uid in self.active_campaign_episodes.keys():
            if uid in active_campaigns:
                reward = 1 * (active_campaigns[uid].budget / active_campaigns[uid].reach)
            else:
                reward = 0
            
            '''
            if self.active_campaign_episodes[uid][3][0] < self.get_current_day():
                profit_change = (self.get_cumulative_profit - self.active_campaign_episodes[uid][3][1]) / self.active_campaign_episodes[uid][3][1]
                reward *= profit_change
                continue
            '''
            
            self.active_campaign_episodes[uid][2] = reward


    '''
    Archives inactive campaigns, and removes old ones if exceeding memory size
    '''
    def archive_yesterday_episodes(self):
        for episode in list(self.active_campaign_episodes.keys()):
            #if self.get_current_day < self.active_campaign_episodes[episode][3][0]:
            #    continue

            self.memory.append(self.active_campaign_episodes[episode])
                
            if len(self.memory) > MEMORY_SIZE:
                self.memory.popleft()
                
            self.active_campaign_episodes.pop(episode)



'''
HYPERPARAMETERS 
'''

# RNN HYPERPARAMS
RNN_NUM_LAYERS = 1
NUM_POSSIBLE_ACTIONS = 50 # different budget subdivisions to bet

# AGENT HYPERPARAMS
NUM_TRAINING_CYCLES = 10
LEARNING_RATE = 0.001
GAMMA = 0.9

EPSILON_START = 0.1
EPSILON_END = 0.05
EPSILON_DECAY = 0.99

BATCH_SIZE = 64
MEMORY_SIZE = 100 # 1 unit is 1 epsiode

IS_TRAINING = True
LOAD_MODEL = True

# OTHER
MODEL_SAVE_PATH = path_from_local_root("latest_model.pth")
