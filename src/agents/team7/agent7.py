import numpy as np
import random
from utils.track_utils import compute_curvature, compute_slope
from agents.kart_agent import KartAgent

class Agent7(KartAgent):
    
    def __init__(self, env, path_lookahead=3):
        """
        Définit les gains du contrôleur (Kp, Kd) et les distances de visée (lookahead)

        Args:
            env (obj): L'environnement de simulation SuperTuxKart
            path_lookahead (int): Nombre de points de cheminement à anticiper (défaut: 3)
        """
        super().__init__(env)
        self.path_lookahead = path_lookahead
        self.name = "Marwane El Moufaoued"
        self.counter = 0.0

        self.Kp = 3
        self.Kd = 1

        # Constante de distance de regard du kart. 
        # Cela va nous permettre de sélectionner les noeuds du circuit devant nous afin de lisser la trajectoire.
        self.ahead_dist = 10

        self.last_error = 0.0   # Contient l'erreur de l'angle précédent 

        self.lookahead_factor = 5
        self.lookahead_max = 14
        self.hairpin_threshold = 0.8
        self.hairpin_accel = 0.2
        self.hairpin_brake_speed = 16

    def reset(self):
        """Réinitialise les variables d'état de l'agent au début d'une course"""
        self.obs, _ = self.env.reset()
        self.last_error = 0.0

    def position_track(self, obs):
        """
        Analyse les noeuds devant et renvoie le vecteur (x, z) du point cible situé à une distance dynamique
        La distance de visée (lookahead) augmente proportionnellement à la vitesse

        Args:
            obs (dict): Dictionnaire contenant les observations de l'environnement

        Returns:
            tuple: (target_vector[0], target_vector[2])
                - target_vector[0] (float): Écart latéral (x) du point cible
                - target_vector[2] (float): Distance frontale (z) du point cible
        """
        # La fonction analyse les noeuds devant et renvoie le vecteur (x, z) du point cible situé à une distance dynamique
        paths = obs['paths_end']

        if len(paths) == 0:
            return 0, self.ahead_dist  # par défaut si aucun noeud n'est donné dans la liste paths_end

        # On calcule la vitesse actuelle pour adapter la distance de visée.
        speed = np.linalg.norm(obs['velocity'])

        # Plus on va vite, plus on regarde loin
        lookahead = self.ahead_dist + (speed * self.lookahead_factor)

        # On plafonne la visée
        lookahead = min(lookahead, self.lookahead_max)

        target_vector = paths[-1]  # Par défaut on prend le noeud le plus loin pour éviter tout bug

        # On cherche le premier point qui dépasse notre distance de visée calculée
        for p in paths:
            if p[2] > lookahead:
                target_vector = p
                break

        # On retourne l'écart latéral x et l'écart avant z du point cible
        return target_vector[0], target_vector[2]

    def position_track_back(self, obs):
        """
        Analyse les noeuds devant et renvoie le vecteur (x, z) du point cible situé à une distance dynamique
        La distance de visée (lookahead) augmente proportionnellement à la vitesse

        Args:
            obs (dict): Dictionnaire contenant les observations de l'environnement

        Returns:
            tuple: (target_vector[0], target_vector[2])
                - target_vector[0] (float): Écart latéral (x) du point cible
                - target_vector[2] (float): Distance frontale (z) du point cible
        """
        # La fonction analyse les noeuds devant et renvoie le vecteur (x, z) du point cible situé à une distance dynamique
        paths = obs['paths_start']

        if len(paths) == 0:
            return 0, self.ahead_dist  # par défaut si aucun noeud n'est donné dans la liste paths_end

        # On calcule la vitesse actuelle pour adapter la distance de visée.
        speed = np.linalg.norm(obs['velocity'])

        # Plus on va vite, plus on regarde loin
        lookahead = self.ahead_dist + (speed * self.lookahead_factor)

        # On plafonne la visée
        lookahead = min(lookahead, self.lookahead_max)

        target_vector = paths[-1]  # Par défaut on prend le noeud le plus loin pour éviter tout bug

        # On cherche le premier point qui dépasse notre distance de visée calculée
        for p in paths:
            if p[2] > lookahead:
                target_vector = p
                break

        # On retourne l'écart latéral x et l'écart avant z du point cible
        return target_vector[0], target_vector[2]

    def compute_turning(self, x, z):
        """
        Calcule l'angle du volant (steering) en fonction des distances (x, z)
        Utilise un gain Proportionnel (Kp) pour la direction et un gain Dérivé (Kd) comme amortisseur
        """
        # La fonction calcule l'angle du volant en fonction des distances (x, z)

        # On évite de diviser par zéro si le point est trop proche
        if z < 1:
            z = 1
        error_angle = x / z  # C'est tan(theta) = opposé/adjacent

        # La dérivée mesure la vitesse à laquelle on corrige l'erreur.
        # Formule = (Erreur de maintenant) - (Erreur d'avant).
        # Elle sert d'amortisseur pour éviter les zigzags.
        error_diff = error_angle - self.last_error
        self.last_error = error_angle

        # Steering = (Force brute vers la cible * Kp) + (Freinage pour pas dépasser * Kd)
        steering = (error_angle * self.Kp) + (error_diff * self.Kd)

        # On limite entre -1 et 1
        steering_normalise = np.clip(steering, -1, 1)

        return steering_normalise




    def manage_speed(self, obs, steering):
        
        """
        
        """
        dist_now = obs['distance_down_track']
        velocity = obs['velocity']
        speed = np.linalg.norm(velocity)

        # On commence par la vitesse d'accélération par défaut configurée
        accel = 0.8
        brake = False

        # Virage standard : on ralentit un peu si le volant dépasse un certain seuil
        if abs(steering) > 0.5:
            accel = 0.3

        # Si le volant est braqué à fond
        if abs(steering) > self.hairpin_threshold:
            # On réduit fortement l'accélération pour permettre au kart de pivoter sur lui-même
            accel = self.hairpin_accel
            # Si on arrive trop vite dans l'épingle, on force un coup de frein
            if speed > self.hairpin_brake_speed:
                brake = True

        return accel, brake, steering

    def choose_action(self, obs):
      
        
        self.counter += 1
        target_x, target_z = self.position_track(obs)
        steering = self.compute_turning(target_x, target_z)
        accel, brake, steering = self.manage_speed(obs, steering)

        if self.counter < 200:
            # si on est inférieur à 200 pas, alors on avance correctement
            

            action = {
            "acceleration": accel,
            "steer": steering,
            "brake": False,
            "drift": False, 
            "nitro": False, 
            "rescue": False,
            "fire": False
        }
        elif 201 < self.counter < 250 :

            # Après 250 pas et supérieur à 200 pas, alors on s'arrête
            action = {
                "acceleration": 0.0,
                "steer": 0.0,
                "brake": True,
                "drift": False, 
                "nitro": False, "rescue": False,
                "fire": False
            }

        else:
            target_x, target_z = self.position_track_back(obs)
            steering = self.compute_turning(target_x, target_z)
            accel, brake, steering = self.manage_speed(obs, steering)
            self.counter = 2000 # on met le compteur à
            action = {
                "acceleration": 0.0,
                "steer": -steering,
                "brake": True,
                "drift": False, 
                "nitro": False, "rescue": False,
                "fire": False
            }
        return action