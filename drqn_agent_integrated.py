from path_utils import path_from_local_root

from drqn_agent import DRQNAgent
from drqn_agent2 import NNModel

import random
import numpy as np

import torch

# build on top of Ad model
class IntegratedModelRL(DRQNAgent):
    def __init__(self, name):
        super().__init__(name)
        
        self.campaign_nn_model = NNModel()
        self.campaign_nn_model.load_state_dict(torch.load(CAMPAIGN_MODEL_PATH))
        
        self.active_campaign_episodes = {}
    
    
    def get_campaign_bids(self, campaigns_for_auction):
        bids = {} # campaign; bid

        for campaign in campaigns_for_auction:
            segment = self.hash_target_segment(campaign.target_segment)
            length = campaign.end_day - campaign.start_day + 1
            
            state = [float(segment), float(campaign.reach), float(length), float(self.get_current_day())]
            
            with torch.no_grad():
                q_values = self.campaign_nn_model(torch.tensor(state).unsqueeze(0))
            
            bid_idx = np.argmax(q_values).item()
            
            if random.random() <= CURR_EPSILON:
                bid_idx = random.randint(0,NUM_POSSIBLE_CAMPAIGN_BIDS-1)
            
            bid_value = max(((bid_idx + 1) / NUM_POSSIBLE_CAMPAIGN_BIDS), 0.1) * campaign.reach 
            
            bids[campaign] = bid_value
        
        return bids


'''
HYPERPARAMETERS
'''

CURR_EPSILON = 0.05
NUM_POSSIBLE_CAMPAIGN_BIDS = 50
CAMPAIGN_MODEL_PATH = path_from_local_root("latest_model.pth")
