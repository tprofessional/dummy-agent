import random
from pacai.agents.capture.capture import CaptureAgent
from pacai.bin.capture import CaptureGameState
from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import reflection
from pacai.core.distance import maze as slow_maze
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
from pacai.core.agentstate import AgentState

def getFeatures(self: CaptureAgent, gameState: CaptureGameState, action):
    maze = lambda a, b: self.getMazeDistance(a, b)

    
    nextGameState: CaptureGameState = gameState.generateSuccessor(self.index, action)

    features = {}
    features["bias"] = 1.0

    # Compute the location of pacman after he takes the action.
    teamate_state: AgentState = gameState.getAgentState((self.index + 2) % 4)
    self_state: AgentState = gameState.getAgentState(self.index)
    team_states: list[AgentState] = [gameState.getAgentState(idx) for idx in self.getTeam(gameState)]
    ops_states: list[AgentState] = [gameState.getAgentState(idx) for idx in self.getOpponents(gameState)]

    ourFood = self.getFoodYouAreDefending(gameState)
    opsFood = self.getFood(gameState)


    ourCapsules = self.getCapsulesYouAreDefending(gameState)
    opsCapsules = self.getCapsules(gameState)

    def is_prey(op: AgentState):
        return (self_state.isBraveGhost() and op.isPacman()) or (self_state.isPacman() and op.isScaredGhost())
    
    def is_preditor(op: AgentState):
        return (self_state.isPacman() and op.isBraveGhost()) or (self_state.isScaredGhost() and op.isPacman())
    
    to_ints = lambda x: (int(x[0]), int(x[1]))
    vec_add = lambda x, y: (x[0] + y[0], x[1] + y[1])

    next_pos = to_ints(nextGameState.getAgentPosition(self.index))
    start_pos = to_ints(self.start_state.getAgentPosition(self.index))
    teamate_pos = to_ints(gameState.getAgentPosition((self.index + 2) % 4))

    prey_dists = sorted([maze(next_pos, to_ints(op.getPosition())) for op in ops_states if is_prey(op)])
    preditor_dists = sorted([maze(next_pos, to_ints(op.getPosition())) for op in ops_states if is_preditor(op)])

    if len(prey_dists) > 0:
        features['near-prey'] = prey_dists[0]

    if len(prey_dists) > 1:
        features['far-prey'] = prey_dists[1]

    if len(preditor_dists) > 0:
        features['near-preditor'] = preditor_dists[0]

    if len(preditor_dists) > 1:
        features['far-preditor'] = preditor_dists[1]

    if len(opsCapsules) != 0:
        min_dist = min([maze(next_pos, cap) for cap in opsCapsules])
        features['min-capsule-distance'] = min_dist

    if len(opsFood.asList()) != 0:
        min_dist = min([maze(next_pos, food) for food in opsFood.asList()])
        features['nearest-food'] = min_dist

    features['dist-from-start'] = min(10, maze(next_pos, start_pos))

    features['dist-from-teamate'] = maze(next_pos, teamate_pos)
    for key in features:
        features[key] /= 10.0

    return features

class DefenseAgentDQN(PacmanQAgent, CaptureAgent):
    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        PacmanQAgent.__init__(self, index, epsilon = 0.1, gamma = 0.5, alpha = 0.0001, numTraining = 150)

        self.startDiscountRate = self.getDiscountRate()
        self.startEpsilon = self.getEpsilon()


        # You might want to initialize weights here.
        self.weights = {
            'dist-from-teamate': 0,
            'dist-from-start': 0,
            }
        self.start_state = None

    def registerInitialState(self, gameState:CaptureGameState):
        """
        This method handles the initial setup of the agent and populates useful fields,
        such as the team the agent is on and the `pacai.core.distanceCalculator.Distancer`.

        IMPORTANT: If this method runs for more than 15 seconds, your agent will time out.
        """
        # get distances from board (maze distance)

        PacmanQAgent.registerInitialState(self, gameState)
        CaptureAgent.registerInitialState(self, gameState)
        self.start_state = gameState
        self.epsilon = self.startEpsilon * max(1 - self.episodesSoFar / self.numTraining, 0)

        print(f"Episodes = {self.episodesSoFar}, Learning Rate = {self.alpha}, Epsilon = {self.epsilon}, Weghts = {self.weights}")


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
        features_dict: dict = getFeatures(self, state, action)
        # print(features_dict)

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
        features_dict: dict = getFeatures(self, state, action)

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
