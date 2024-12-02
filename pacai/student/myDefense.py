from pacai.agents.capture.reflex import ReflexCaptureAgent
from pacai.core.directions import Directions

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that tries to keep its side Pacman-free.
    This is to give you an idea of what a defensive agent could be like.
    It is not the best or only way to make such an agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    
    def getFeatures(self, gameState, action):
        features = {}

        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        successorGameState = myState.generateSuccessor(self.index, action)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0).
        features['onDefense'] = 1
        if (myState.isPacman()):
            features['onDefense'] = 0

        # Computes distance to invaders we can see.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)

        if (len(invaders) > 0):
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        # check if our action is STOP
        if (action == Directions.STOP):
            features['stop'] = 1

        # check if our action is reverse
        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1

        # compute number of food we're defending using successor
        numFood = len(successor.getFoodYouAreDefending(successorGameState).asList())
        features['ourFood'] = numFood

        # compute number of food we're defending using successor
        numCapsules = len(successor.getCapsulesYouAreDefending(successorGameState).asList())
        features['ourCapsules'] = numCapsules

        return features

    def getWeights(self, gameState, action):
        return {
            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': -10,
            'stop': -100,
            'reverse': -2,
            'ourFood': 200,
            'ourCapsules': 250
        }
