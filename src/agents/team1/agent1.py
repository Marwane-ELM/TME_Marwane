from pathlib import Path
import numpy as np
import random
from utils.track_utils import compute_curvature, compute_slope
from omegaconf import OmegaConf

from agents.kart_agent import KartAgent
from agents.team1.agent_center import AgentCenter
from agents.team1.agent_speed import AgentSpeed
from agents.team1.agent_obstacles import AgentObstacles
from agents.team1.agent_rescue import AgentRescue
from agents.team1.agent_items import AgentItems
from agents.team1.agent_drift import AgentDrift

class Agent1(KartAgent):
    def __init__(self, env, path_lookahead=3):
        super().__init__(env)
        self.path_lookahead = path_lookahead
        self.agent_positions = []
        self.obs = None
        self.isEnd = False
        self.name = "Marwane El Moufaoued"
        self.counter = 0.0
        self.counter_brake = 0.0

        path_conf = Path(__file__).resolve().parent
        path_conf = str(path_conf) + '/ConfigFileTeam1.yaml'   #Chemin du fichier de configuration
        self.conf = OmegaConf.load(path_conf)                           #Importation du fichier de configuration

        self.agentCenter = AgentCenter(env, self.conf, self.path_lookahead)
        #self.agentSpeed = AgentSpeed(env, self.conf, self.agentCenter, self.path_lookahead)
        #self.agentObstacles = AgentObstacles(env, self.conf, self.agentCenter, self.path_lookahead)
        #self.agentRescue = AgentRescue(env, self.conf, self.agentObstacles)
        #self.agentItems = AgentItems(env, self.conf, self.agentRescue)
        #self.AgentDrift = AgentDrift(env, self.conf, self.agentItems)

    def reset(self):
        self.obs, _ = self.env.reset()
        self.agent_positions = []
        self.counter = 0.0
        self.counter_brake = 0.0

    def endOfTrack(self):
        return self.isEnd


    def choose_action(self, obs):


        # Tant que le compteur est inférieur à 50 frames, alors on tourne à droite
        self.counter += 1
        if(self.counter < 50):
            action = {
                "acceleration": 0.3,
                "steer": 1,
                "brake": False,
                "drift": False,
                "nitro": False,
                "rescue": False,
                "fire": False,
            }
        
        # Puis on s'arrête pendant 10 frames
        elif self.counter <= 60:
            action = {
                "acceleration": 0.0,
                "steer": 0,
                "brake": True,
                "drift": False,
                "nitro": False,
                "rescue": False,
                "fire": False,
            }

        # Avant de partir en marche arrière
        else : 
            self.counter = 2000  # On le bloque à 2000 pour que les 'if' au dessus ne s'activent plus
            action = self.agentCenter.choose_action(obs)

        return action