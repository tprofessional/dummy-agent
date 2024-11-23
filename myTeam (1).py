import random
from pacai.agents.capture.capture import CaptureAgent
from pacai.core.distance import manhattan
from pacai.bin.capture import CaptureGameState

from pacai.student.defensiveAgents import *

def createTeam(firstIndex, secondIndex, isRed):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """
    
    firstAgent = DummyAgent1
    secondAgent = DummyAgent2

    return [
        firstAgent(firstIndex),
        secondAgent(secondIndex),
    ]

# Copied over from pacai/agents/capture/dummy.py
# survive function for pacman 
class OffenseAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at `pacai.core.baselineTeam` for more details about how to create an agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
    
    def registerInitialState(self, gameState:CaptureGameState):
        """
        This method handles the initial setup of the agent and populates useful fields,
        such as the team the agent is on and the `pacai.core.distanceCalculator.Distancer`.

        IMPORTANT: If this method runs for more than 15 seconds, your agent will time out.
        """
        # get distances from board (maze distance)

        super().registerInitialState(gameState)

        # Your initialization code goes here, if you need any.

    def chooseAction(self, gameState):
        """
        Pick an action.
        """
        # get state and see if we're on the right or left of center line
        # var ghost
        # if ghost, be defensive
        # if pacman, be offensive
        actions = gameState.getLegalActions(self.index)
        return random.choice(actions)
    
    def betterEvaluationFunction(currentGameState):
        """
        Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

        DESCRIPTION: I rewrote this function to be less of a reflex agent and more of a function
        that keeps track of the overall state of the game. This is why I removed the for loop
        which goes through each ghost's scared time and instead opted for a more big-picture
        approach where I considered the minimum of all scared times instead. By adding this value to
        the score, the pacman would be incentivized to be in a state where the scared times for all
        the ghosts are greater rather than smaller. Similarly, I incentivized eating more food by
        subtracting the amount of food left in the state from the score.
        """

        # get PM's position in successor state
        currPosition = currentGameState.getPacmanPosition()
        # get list of food
        foods = currentGameState.getFood()
        # get state of ghosts in successor state (list)
        ghostStates = currentGameState.getGhostStates()
        # get scared timers for ghosts
        scaredTimes = [ghostState.getScaredTimer() for ghostState in ghostStates]

        # init score as the score of successor state
        score = currentGameState.getScore()

        # we want to bring up all the scared times
        score += min(scaredTimes)

        # manhattan dist to every food
        foodDistances = [manhattan(currPosition, food) for food in foods.asList()]

        # incentivize moving twd food
        if foodDistances:
            closestFood = min(foodDistances)
            score -= closestFood

        # want to incentivize eating all the food
        score -= len(foods.asList())

        return score

# Copied over from pacai/agents/capture/dummy.py
# ghost heavily incentivize chasing pacman
# for registerInitialState keep track of how close opp pacman is to ghost pellet
class DefenseAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at `pacai.core.baselineTeam` for more details about how to create an agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the agent and populates useful fields,
        such as the team the agent is on and the `pacai.core.distanceCalculator.Distancer`.

        IMPORTANT: If this method runs for more than 15 seconds, your agent will time out.
        """

        super().registerInitialState(gameState)

        # Your initialization code goes here, if you need any.

    def chooseAction(self, gameState):
        """
        Randomly pick an action.
        """

        actions = gameState.getLegalActions(self.index)
        return random.choice(actions)
