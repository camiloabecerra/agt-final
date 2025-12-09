from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment 
from path_utils import path_from_local_root
from typing import Set, Dict

from drqn_agent import DRQNAgent
import numpy as np
import torch

class MyNDaysNCampaignsAgent(DRQNAgent):

    def __init__(self):
        # TODO: fill this in (if necessary)
        name = "MegaKnight"
        
        super().__init__(name)
        self.name = name


    def on_new_game(self) -> None:
        # TODO: fill this in (if necessary)
        path = path_from_local_root("latest_model.pth")

        model = RNNModel()
        
        model.load_state_dict(torch.load(path))
        self.impression_rnn_model = model


    def get_campaign_bids(self, campaigns_for_auction:  Set[Campaign]) -> Dict[Campaign, float]:
        # TODO: fill this in 
        bids = {} # campaign; bid
        our_campaigns = self.get_active_campaigns()
        
        our_budgets = []
        our_targets = set()
        for our_campaign in our_campaigns:
            our_budgets.append(our_campaign.budget)
            our_targets.add(our_campaign.target_segment)
        
        avg_price = np.mean(our_budgets)

        for campaign in campaigns_for_auction:
            included = False
            
            for target in our_targets:
                included |= target.issubset(campaign.target_segment)
                included |= campaign.target_segment.issubset(target)
            
            if not included and not (campaign.target_segment in our_targets):
                bids[campaign] = avg_price
            else:
                bids[campaign] = 0
        
        return bids


if __name__ == "__main__":
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    test_agents = [MyNDaysNCampaignsAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]
    
    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=500)
