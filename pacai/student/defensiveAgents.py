import random
from pacai.agents.capture.capture import CaptureAgent
from pacai.core.distance import maze
from pacai.bin.capture import CaptureGameState
from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import reflection
from pacai.student.qlearningAgents import PacmanQAgent

import random
import logging

import time
from pacai.util import util

# Option for the defensive agent that uses a deep q network (approximate q function)
# that learns from a set of extracted features to act in a way that maximises short
# and expected long term rewards
#
# This class needs a feature extractor and a custom reward function that incentivises
# ceratin game states

from pacai.core.actions import Actions
from pacai.core.search import search
from pacai.student.searchAgents import AnyFoodSearchProblem

def getFeatures(state, action):
    print(type(state), state)

    # Extract the grid of food and wall locations and get the ghost locations.
    food = state.getFood()
    walls = state.getWalls()
    ghosts = state.getGhostPositions()

    features = {}
    features["bias"] = 1.0

    # Compute the location of pacman after he takes the action.
    x, y = state.getPacmanPosition()
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(x + dx), int(y + dy)

    # Count the number of ghosts 1-step away.
    features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in
            Actions.getLegalNeighbors(g, walls) for g in ghosts)

    # If there is no danger of ghosts then add the food feature.
    if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
        features["eats-food"] = 1.0

    prob = AnyFoodSearchProblem(state, start = (next_x, next_y))
    dist = len(search.bfs(prob))
    if dist is not None:
        # Make the distance a number less than one otherwise the update will diverge wildly.
        features["closest-food"] = float(dist) / (walls.getWidth() * walls.getHeight())

    for key in features:
        features[key] /= 10.0

    return features

class DefenseAgentDQN(PacmanQAgent, CaptureAgent):
    def __init__(self, index, **kwargs):
        
        print(kwargs)
        super().__init__(index, **kwargs)

        # You might want to initialize weights here.
        self.weights = {}


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
        features_dict: dict = getFeatures(state, action)

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
        features_dict: dict = getFeatures(state, action)

        for feature, value in features_dict.items():
            self.weights[feature] = self.get_weight(feature) + alpha * correction * value

    def chooseAction(self, gameState):
        action = super().getPolicy(gameState)
        print(action)
        return action



# Copied over from pacai/agents/capture/dummy.py 
class DummyAgent1(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at `pacai.core.baselineTeam` for more details about how to create an agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest return from `ReflexCaptureAgent.evaluate`.
        """

        actions = gameState.getLegalActions(self.index)

        start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        logging.debug('evaluate() time for agent %d: %.4f' % (self.index, time.time() - start))

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """

        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()

        if (pos != util.nearestPoint(pos)):
            # Only half a grid position was covered.
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights.
        """

        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        stateEval = sum(features[feature] * weights[feature] for feature in features)

        return stateEval

    def getFeatures(self, gameState, action):
        features = {}
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)

        # Compute distance to the nearest food.
        foodList = self.getFood(successor).asList()

        # This should always be True, but better safe than sorry.
        if (len(foodList) > 0):
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        return features

    def getWeights(self, gameState, action):
        return {
            'successorScore': 100,
            'distanceToFood': -1
        }
