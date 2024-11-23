import random
from pacai.agents.capture.capture import CaptureAgent
from pacai.core.distance import maze
from pacai.bin.capture import CaptureGameState
from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import reflection
from pacai.student.qlearningAgents import PacmanQAgent

# Option for the defensive agent that uses a deep q network (approximate q function)
# that learns from a set of extracted features to act in a way that maximises short
# and expected long term rewards
#
# This class needs a feature extractor and a custom reward function that incentivises
# ceratin game states

class DefenseAgentDQN(PacmanQAgent):
    def __init__(self, index,
            extractor = 'pacai.core.featureExtractors.IdentityExtractor', **kwargs):
        super().__init__(index, **kwargs)
        self.featExtractor = reflection.qualifiedImport(extractor)

        # You might want to initialize weights here.
        self.weights = {}
        self.extractor = self.featExtractor()

    def final(self, state):
        """
        Called at the end of each game.
        """

        # Call the super-class final method.
        super().final(state)

        # Did we finish training?
        if self.episodesSoFar == self.numTraining:
            # You might want to print your weights here for debugging.
            # *** Your Code Here ***
            print(self.weights)
        
    def get_weight(self, feature):
        return self.weights.get(feature, 0.0)

    def getQValue(self, state, action):
        features_dict: dict = self.extractor.getFeatures(state, action)

        QValue = 0.0
        for feature, value in features_dict.items():
            QValue += self.get_weight(feature) * value

        return QValue
    
    def update(self, state, action, nextState, reward):
        gamma = self.getGamma()
        nextValue = self.getValue(nextState)
        qValue = self.getQValue(state, action)
        correction = reward + gamma * nextValue - qValue

        alpha = self.getAlpha()
        features_dict: dict = self.extractor.getFeatures(state, action)

        for feature, value in features_dict.items():
            self.weights[feature] = self.get_weight(feature) + alpha * correction * value