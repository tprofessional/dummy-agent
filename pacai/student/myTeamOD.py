import random
import logging

import time
from pacai.util import util
from pacai.agents.capture.capture import CaptureAgent
from pacai.core.distance import manhattan
from pacai.core.directions import Directions


from pacai.bin.capture import CaptureGameState
from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import reflection
from pacai.core.distance import maze as slow_maze
from pacai.student.qlearningAgents import PacmanQAgent
from pacai.agents.capture.reflex import ReflexCaptureAgent



def createTeam(firstIndex, secondIndex, isRed):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    firstAgent = DummyAgent1
    secondAgent = DefensiveReflexAgent

    return [
        firstAgent(firstIndex),
        secondAgent(secondIndex),
    ]

class DummyAgent1(CaptureAgent):
    """
    A Dummy agent that prioritizes food collection while maintaining
    a safe distance from ghosts (at least 3 steps), now using alpha-beta pruning.
    """

    def chooseAction(self, gameState):
        """
        Choose the best action using alpha-beta pruning.
        """
        # Initialize alpha and beta
        alpha = float('-inf')
        beta = float('inf')
        depth = 3  # Depth limit for the search

        # Perform alpha-beta pruning to find the best action
        bestAction = None
        bestValue = float('-inf')

        for action in gameState.getLegalActions(self.index):
            if action == "Stop":
                continue

            # Evaluate the successor state
            successor = self.getSuccessor(gameState, action)
            value = self.alphaBeta(successor, depth - 1, alpha, beta, isMaximizing=False)

            if value > bestValue:
                bestValue = value
                bestAction = action

            # Update alpha
            alpha = max(alpha, bestValue)

        return bestAction

    def alphaBeta(self, gameState, depth, alpha, beta, isMaximizing):
        """
        Alpha-beta pruning algorithm.
        """
        if depth == 0 or gameState.isOver():
            return self.evaluate(gameState, None)

        actions = gameState.getLegalActions(self.index)

        if isMaximizing:
            value = float('-inf')
            for action in actions:
                if action == "Stop":
                    continue
                successor = self.getSuccessor(gameState, action)
                value = max(value, self.alphaBeta(successor, depth - 1, alpha, beta, isMaximizing=False))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break 
            return value
        else:
            value = float('inf')
            
            opponents = self.getOpponents(gameState)
            for opponent in opponents:
                for action in gameState.getLegalActions(opponent):
                    successor = gameState.generateSuccessor(opponent, action)
                    value = min(value, self.alphaBeta(successor, depth - 1, alpha, beta, isMaximizing=True))
                    beta = min(beta, value)
                    if beta <= alpha:
                        break  
            return value

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
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
        successor = self.getSuccessor(gameState, action) if action else gameState
        features['successorScore'] = self.getScore(successor)

        # Compute distance to the nearest food
        foodList = self.getFood(successor).asList()

        if len(foodList) > 0:
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        opponent_indices = self.getOpponents(gameState)
        opponent_states = [gameState.getAgentState(index) for index in opponent_indices]
    
        
        ghost_positions = [opponent.getPosition() for opponent in opponent_states if not opponent.isPacman() and opponent.getPosition() is not None]
    

        if len(ghost_positions) > 0:
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, ghost) for ghost in ghost_positions])
            features['distanceToGhost'] = minDistance
        else:
            features['distanceToGhost'] = 0
            
        return features

    def getWeights(self, gameState, action):
        return {
            'successorScore': 100,
            'distanceToFood': -1,
            'distanceToGhost': 0.5
        }


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that tries to keep its side Pacman-free.
    This is to give you an idea of what a defensive agent could be like.
    It is not the best or only way to make such an agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)
    
    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest return from `ReflexCaptureAgent.evaluate`.
        """

        actions = gameState.getLegalActions(self.index)

        values = [self.evaluate(gameState, a) for a in actions]

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        return random.choice(bestActions)
    
    def alphaBeta(self, gameState, depth, alpha, beta, isMaximizing):
        """
        Alpha-beta pruning algorithm.
        """
        if depth == 0 or gameState.isOver():
            return self.evaluate(gameState, None)

        actions = gameState.getLegalActions(self.index)

        if isMaximizing:
            value = float('-inf')
            for action in actions:
                if action == "Stop":
                    continue
                successor = self.getSuccessor(gameState, action)
                value = max(value, self.alphaBeta(successor, depth - 1, alpha, beta, isMaximizing=False))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break 
            return value
        else:
            value = float('inf')
            
            opponents = self.getOpponents(gameState)
            for opponent in opponents:
                for action in gameState.getLegalActions(opponent):
                    successor = gameState.generateSuccessor(opponent, action)
                    value = min(value, self.alphaBeta(successor, depth - 1, alpha, beta, isMaximizing=True))
                    beta = min(beta, value)
                    if beta <= alpha:
                        break  
            return value
    
    def getFeatures(self, gameState, action):
        features = {}

        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        successorGameState = gameState.generateSuccessor(self.index, action)
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
        numFood = len(self.getFoodYouAreDefending(successorGameState).asList())
        features['ourFood'] = numFood

        # compute number of food we're defending using successor
        numCapsules = len(self.getCapsulesYouAreDefending(successorGameState))
        features['ourCapsules'] = numCapsules

        # compute scaredTimer
        # if it's greater than #
            # and we're close to the opp & opp is 
                # insta die
            # elif we're close to the opp side 
                # cross the border --> this is where we can switch to offense
                # might not implement this rn

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
    