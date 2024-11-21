import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.bin.pacman import PacmanGameState

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Useful information you can extract.
        # newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        # *** Your Code Here ***
        manhattan = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])

        newPos = successorGameState.getPacmanPosition()
        newGhostStates = successorGameState.getGhostStates()

        minGhostDist = float('inf')
        for newGhostPos in [_.getPosition() for _ in newGhostStates]:
            minGhostDist = min(minGhostDist, manhattan(newPos, newGhostPos))

        minFoodDist = float('inf')
        for food in currentGameState.getFood().asList():
            # minFoodDist = min(minFoodDist, maze(newPos, food, currentGameState))
            minFoodDist = min(minFoodDist, manhattan(newPos, food))

        return successorGameState.getScore() + min(minGhostDist, 3) * 9999 - minFoodDist

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    from pacai.bin.pacman import PacmanGameState

    def getAction(self, state: PacmanGameState):
        actions = state.getLegalActions(0)
        actions.remove('Stop')
        bestAction = 'Stop'
        bestScore = float('-inf')

        for action in actions:
            newState = state.generateSuccessor(0, action)
            newScore = self.minmax(newState, self.getTreeDepth(), 1)
            if newScore > bestScore:
                bestScore = newScore
                bestAction = action

        return bestAction

    def minmax(self, state: PacmanGameState, depth, player):
        if depth == 0 or not state.getLegalActions(player):
            return self.getEvaluationFunction()(state)
        
        actions = state.getLegalActions(player)
        if 'Stop' in actions:
            actions.remove('Stop')

        if player == 0:     # pacman's move
            bestScore = float('-inf')
            for action in actions:
                newState = state.generateSuccessor(player, action)
                newScore = self.minmax(newState, depth, 1)
                bestScore = max(bestScore, newScore)
            return bestScore
            
        else:   # ghosts moves
            bestScore = float('inf')
            nextPlayer = (player + 1) % state.getNumAgents()
            for action in actions:
                newState = state.generateSuccessor(player, action)
                if nextPlayer == 0:
                    newScore = self.minmax(newState, depth - 1, nextPlayer)
                else:
                    newScore = self.minmax(newState, depth, nextPlayer)
                bestScore = min(bestScore, newScore)
            return bestScore
            

class AlphaBetaAgent(MultiAgentSearchAgent):

    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    from pacai.bin.pacman import PacmanGameState

    def getAction(self, state: PacmanGameState):
        actions = state.getLegalActions(0)
        actions.remove('Stop')
        bestAction = 'Stop'
        bestScore = float('-inf')

        for action in actions:
            newState = state.generateSuccessor(0, action)
            newScore = self.alphaBeta(newState, self.getTreeDepth(), 1, float('-inf'), float('inf'))
            if newScore > bestScore:
                bestScore = newScore
                bestAction = action

        return bestAction

    def alphaBeta(self, state, depth, player, alpha, beta):
        if depth == 0 or not state.getLegalActions(player):
            return self.getEvaluationFunction()(state)

        actions = state.getLegalActions(player)
        if 'Stop' in actions:
            actions.remove('Stop')

        if player > 0 and not actions:
            nextPlayer = (player + 1) % state.getNumAgents()
            if nextPlayer == 0:
                return self.alphaBeta(state, depth - 1, nextPlayer, alpha, beta)
            else:
                return self.alphaBeta(state, depth, nextPlayer, alpha, beta)

        if player == 0:  # Pacman's move (maximize score)
            bestScore = float('-inf')
            for action in actions:
                newState = state.generateSuccessor(player, action)
                newScore = self.alphaBeta(newState, depth - 1, 1, alpha, beta)
                bestScore = max(bestScore, newScore)
                alpha = max(alpha, bestScore)
                if beta <= alpha:
                    break
            return bestScore

        else:  # Ghost's move (minimize score)
            bestScore = float('inf')
            nextPlayer = (player + 1) % state.getNumAgents()
            for action in actions:
                newState = state.generateSuccessor(player, action)
                if nextPlayer == 0:
                    newScore = self.alphaBeta(newState, depth - 1, nextPlayer, alpha, beta)
                else:
                    newScore = self.alphaBeta(newState, depth, nextPlayer, alpha, beta)
                bestScore = min(bestScore, newScore)
                beta = min(beta, bestScore)
                if beta <= alpha:
                    break
            return bestScore

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    from pacai.bin.pacman import PacmanGameState

    def getAction(self, state: PacmanGameState):
        actions = state.getLegalActions(0)
        actions.remove('Stop')
        bestAction = 'Stop'
        bestScore = float('-inf')

        for action in actions:
            newState = state.generateSuccessor(0, action)
            newScore = self.expectimax(newState, self.getTreeDepth(), 1)
            if newScore > bestScore:
                bestScore = newScore
                bestAction = action

        return bestAction

    def expectimax(self, state: PacmanGameState, depth, player):
        if depth == 0 or not state.getLegalActions(player):
            return self.getEvaluationFunction()(state)
        
        actions = state.getLegalActions(player)
        if 'Stop' in actions:
            actions.remove('Stop')

        if player == 0:     # pacman's move
            bestScore = float('-inf')
            for action in actions:
                newState = state.generateSuccessor(player, action)
                newScore = self.expectimax(newState, depth, 1)
                bestScore = max(bestScore, newScore)
            return bestScore
            
        else:   # ghosts moves
            avgScore = 0
            nextPlayer = (player + 1) % state.getNumAgents()
            for action in actions:
                newState = state.generateSuccessor(player, action)
                if nextPlayer == 0:
                    newScore = self.expectimax(newState, depth - 1, nextPlayer)
                else:
                    newScore = self.expectimax(newState, depth, nextPlayer)
                avgScore += newScore
            avgScore /= max(1, len(actions))
            return avgScore
        
def betterEvaluationFunction(currentGameState: PacmanGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>

    My code computes a sum of different bonuses that the agent can get from:
    -Distance to nearest food: add the nearest food's maze distance * -0.5 to encourage
    moving to the nearest food

    -Distance to capsule: add sum of inverse manhattan distacnes to capsules if no
    ghosts are scared and there are still capsules on the board. Otherwise give a flat reward
    of 100

    -Distance to scared ghosts: add sum of 200 * inverse manhattan distances to scared ghosts

    returns gameScore + foodScore + capsuleScore + ghostScore
    """
    from pacai.core.distance import maze

    manhattan = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])

    pos = currentGameState.getPacmanPosition()
    foodPositions = currentGameState.getFood().asList()
    ghostPositions = [_.getPosition() for _ in currentGameState.getGhostStates()]
    ghostScaredTimes = [_.getScaredTimer() for _ in currentGameState.getGhostStates()]
    capsulePositions = currentGameState.getCapsules()

    minGhostDist = float('inf')
    for ghostPos in ghostPositions:
        minGhostDist = min(minGhostDist, manhattan(pos, ghostPos))

    # Get the manhattan distance to all food
    foodMan = [(food, manhattan(pos, food)) for food in foodPositions]
    foodMan.sort(key=lambda x: x[1])

    # Compute the maze distance to the nearest 2 manhattan distances
    foodMaze = [maze(pos, food[0], currentGameState) for food in foodMan[:2]]
    foodMaze.sort()
    
    foodScore = -min(foodMaze) if len(foodMaze) > 0 else 0
    # needs to make sure that when remaining food is near then far the increase
    # in loss wont be an issue

    # Get the inverse distance to all capsules
    # Agent should go towards capsules if ghosts are scared
    capsuleInvMan = [1 / max(manhattan(pos, cap), 1) for cap in capsulePositions]
    capsuleScore = sum(capsuleInvMan) if len(capsuleInvMan) > 0 else 100
    if max(ghostScaredTimes) > 2:
        capsuleScore = 100
    
    # Prioritize chasing ghosts when they are scared
    ghostScore = 0
    for ghost, scaredTime in zip(ghostPositions, ghostScaredTimes):
        if scaredTime > 2:
            ghostScore += 200 / max(manhattan(pos, ghost), 1)

    return currentGameState.getScore() + foodScore * 0.5 + capsuleScore + ghostScore

def betterEvaluationFunctionCheap(currentGameState: PacmanGameState):
    """
    My code computes a sum of different bonuses that the agent can get from:
    -Distance to nearest food: add the nearest food's maze distance * -0.5 to encourage
    moving to the nearest food

    -Distance to capsule: add sum of inverse manhattan distacnes to capsules if no
    ghosts are scared and there are still capsules on the board. Otherwise give a flat reward
    of 100

    -Distance to scared ghosts: add sum of 200 * inverse manhattan distances to scared ghosts

    returns gameScore + foodScore + capsuleScore + ghostScore
    """
    
    manhattan = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])

    pos = currentGameState.getPacmanPosition()
    foodPositions = currentGameState.getFood().asList()
    ghostPositions = [_.getPosition() for _ in currentGameState.getGhostStates()]
    ghostScaredTimes = [_.getScaredTimer() for _ in currentGameState.getGhostStates()]
    capsulePositions = currentGameState.getCapsules()

    minGhostDist = float('inf')
    for ghostPos in ghostPositions:
        minGhostDist = min(minGhostDist, manhattan(pos, ghostPos))

    # Get the manhattan distance to all food
    foodMan = [manhattan(pos, food) for food in foodPositions]
    foodScore = -min(foodMan) if len(foodMan) > 0 else 0
    # needs to make sure that when remaining food is near then far the increase
    # in loss wont be an issue

    # Get the inverse distance to all capsules
    # Agent should go towards capsules if ghosts are scared
    capsuleInvMan = [1 / max(manhattan(pos, cap), 1) for cap in capsulePositions]
    capsuleScore = sum(capsuleInvMan) if len(capsuleInvMan) > 0 else 100
    if max(ghostScaredTimes) > 2:
        capsuleScore = 100 + 100 * len(capsuleInvMan)
    
    # Prioritize chasing ghosts when they are scared
    ghostScore = 0
    for ghost, scaredTime in zip(ghostPositions, ghostScaredTimes):
        if scaredTime > 2:
            ghostScore += 200 / max(manhattan(pos, ghost), 1)

    return currentGameState.getScore() + foodScore * 0.5 + capsuleScore + ghostScore

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        self._evaluationFunction = betterEvaluationFunctionCheap
        self._treeDepth = 3

    from pacai.bin.pacman import PacmanGameState

    def getAction(self, state: PacmanGameState):
        actions = state.getLegalActions(0)
        actions.remove('Stop')
        bestAction = 'Stop'
        bestScore = float('-inf')

        for action in actions:
            newState = state.generateSuccessor(0, action)
            newScore = self.expectimax(newState, self.getTreeDepth(), 1)
            if newScore > bestScore:
                bestScore = newScore
                bestAction = action

        return bestAction

    def expectimax(self, state: PacmanGameState, depth, player):
        if depth == 0 or not state.getLegalActions(player):
            return self.getEvaluationFunction()(state)
        
        actions = state.getLegalActions(player)
        if 'Stop' in actions:
            actions.remove('Stop')

        if player == 0:     # pacman's move
            bestScore = float('-inf')
            for action in actions:
                newState = state.generateSuccessor(player, action)
                newScore = self.expectimax(newState, depth, 1)
                bestScore = max(bestScore, newScore)
            return bestScore
            
        else:   # ghosts moves
            avgScore = 0
            nextPlayer = (player + 1) % state.getNumAgents()
            for action in actions:
                newState = state.generateSuccessor(player, action)
                if nextPlayer == 0:
                    newScore = self.expectimax(newState, depth - 1, nextPlayer)
                else:
                    newScore = self.expectimax(newState, depth, nextPlayer)
                avgScore += newScore
            avgScore /= max(1, len(actions))
            return avgScore