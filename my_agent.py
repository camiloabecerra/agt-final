from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment 
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
        pass


    def get_campaign_bids(self, campaigns_for_auction:  Set[Campaign]) -> Dict[Campaign, float]:
        # TODO: fill this in 
        bids = {} # campaign; bid
        our_campaigns = self.get_active_campaigns()
        
        our_budgets = []
        our_targets = set()
        for our_campaign in our_campaigns:
            our_budgets.append(our_campaign.budget)
            our_targets.add(our_campaign.target_segment)
        
        
        for campaign in campaigns_for_auction:
            value_per_impression = 1

            included = False
            for target in our_targets:
                included |= target.issubset(campaign.target_segment)
                included |= campaign.target_segment.issubset(target)

            if included:
                value_per_impression *= 1.5
            elif campaign.target_segment in our_targets:
                value_per_impression *= 1.25
            
            length = campaign.end_day - campaign.start_day + 1
            
            value_per_impression *= ((1/(length)) + 0.75)
            
            seg = self.hash_target_segment(campaign.target_segment)
            if seg > 11:
                value_per_impression *= 1.3
            else:
                value_per_impression *= 0.8

            
            if value_per_impression != 0:
                bids[campaign] = value_per_impression * campaign.reach
            
        return bids


if __name__ == "__main__":
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    test_agents = [MyNDaysNCampaignsAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]
    
    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=500)
