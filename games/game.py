from abc import abstractmethod

try:
    from player import Player
except:
    from games.player import Player

class Game():
    def __init__(self, name, players):
        self.name = name
        self.n_players = len(players)
        self.n_actions = [player.n_actions for player in players]
        self.players = players  
        
    @abstractmethod
    def find_nash(self, *args, **kwargs):
        pass
