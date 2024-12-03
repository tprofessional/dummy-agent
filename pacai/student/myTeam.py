from pacai.agents.capture.capture import CaptureAgent
from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.bin.capture import CaptureGameState
from pacai.core.agentstate import AgentState
from pacai.core.gamestate import AbstractGameState
from pacai.util.probability import flipCoin
from pacai.core.directions import Directions
from random import choice

def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.student.defensiveAgents.DummyAgent1',
        second = 'pacai.student.defensiveAgents.DummyAgent1'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    firstAgent = CaptureQAgent(
        firstIndex,
        feature_extractor=betterFeatures,
        weights={'defense-target-dist': -1, 'on-wrong-side': -1000,
                 'food-score': 100, 'caps-score': 10, 'near-ghost': 0,
                 'stop': -2, 'food-needs-defense': -10, 'kills-target': 100,
                 'dies': -1000, 'ghost-1-away': -1000},
        debug=False
    )

    secondAgent = CaptureQAgent(
        secondIndex,
        feature_extractor=betterFeatures,
        weights={'defense-target-dist': -1, 'on-wrong-side': -1000,
                 'food-score': 100, 'caps-score': 10, 'near-ghost': 0,
                 'stop': -2, 'food-needs-defense': -10, 'kills-target': 100,
                 'dies': -1000, 'ghost-1-away': -1000},
        debug=False
    )

    return [firstAgent, secondAgent]

# Option for the defensive agent that uses a deep q network (approximate q function)
# that learns from a set of extracted features to act in a way that maximises short
# and expected long term rewards
#
# This class needs a feature extractor and a custom reward function that incentivises
# ceratin game states

def betterFeatures(self: CaptureAgent, state: CaptureGameState, action):
    to_ints = lambda x: (int(x[0]), int(x[1]))
    maze = lambda a, b: self.getMazeDistance(to_ints(a), to_ints(b))
    inv_exp_sum = lambda arr: sum([0.33 ** x for x in arr])

    features = {}
    features["bias"] = 1.0

    # Compute the location of pacman after he takes the action.
    nextState: CaptureGameState = state.generateSuccessor(self.index, action)
    nextSelfState: AgentState = nextState.getAgentState(self.index)
    opsTeam: list[AgentState] = [state.getAgentState(idx) for idx in self.getOpponents(state)]

    def ghost_pred(op: AgentState):
        return nextSelfState.isPacman() and op.isBraveGhost()

    # pos0 = self.start_state.getAgentPosition(self.index)

    pos = state.getAgentPosition(self.index)
    teammatePos = state.getAgentPosition((self.index + 2) % 4)

    nextPos = nextState.getAgentPosition(self.index)

    foodDists = [maze(nextPos, food) for food in self.getFood(nextState).asList()]
    totalFoodEaten = \
        len(self.getFood(self.start_state).asList()) - len(self.getFood(nextState).asList())
    capsDists = [maze(nextPos, cap) for cap in self.getCapsules(state)]
    totalCapsEaten = \
        len(self.getCapsules(self.start_state)) - len(self.getCapsules(nextState))

    ghostDists = [maze(nextPos, op.getPosition()) for op in opsTeam if ghost_pred(op)]

    # targets determined to minimize total distance.
    targetIndex = (self.index + 1) % 4
    teamateTargetIndex = (self.index + 3) % 4
    targetPos = state.getAgentPosition(targetIndex)
    teamateTargetPos = state.getAgentPosition(teamateTargetIndex)

    distanceSum = maze(pos, targetPos) + maze(teammatePos, teamateTargetPos)
    swappedDistanceSum = maze(pos, teamateTargetPos) + maze(teammatePos, targetPos)
    if (distanceSum > swappedDistanceSum):
        targetIndex = (self.index + 3) % 4          # swap targets if it makes sense
        teamateTargetIndex = (self.index + 1) % 4
    targetPos = state.getAgentPosition(targetIndex)
    teamateTargetPos = state.getAgentPosition(teamateTargetIndex)

    targetDist = maze(nextPos, targetPos)
    targetState: AgentState = state.getAgentState(targetIndex)
    # teamateTargetDist = maze(teammatePos, teamateTargetPos)

    foodNeedsDefense = False
    for food in self.getFoodYouAreDefending(nextState).asList():
        if maze(targetPos, food) <= maze(nextPos, food):
            foodNeedsDefense = True
    
    defensiveMode = nextSelfState.isBraveGhost()
    if defensiveMode:
        features['defense-target-dist'] = maze(nextPos, targetPos)
        features['on-wrong-side'] = \
            nextState.isOnRedSide(nextPos) != nextState.isOnRedTeam(self.index)
    else:
        features['food-score'] = totalFoodEaten + inv_exp_sum(foodDists)
        features['caps-score'] = totalCapsEaten + 1 / max(min(capsDists, default=1), 1)
        features['near-ghost'] = min(ghostDists, default=0)
        features['reverse'] = (
            action
            == Directions.REVERSE[state.getAgentState(self.index).getDirection()]
        )
    features['stop'] = action == Directions.STOP
    features['food-needs-defense'] = foodNeedsDefense
    features['kills-target'] = targetDist == 0 and \
        (nextSelfState.isBraveGhost() or targetState.isScaredGhost())
    features['dies'] = maze(pos, nextPos) > 1
    features['ghost-1-away'] = min(ghostDists, default=0) == 1
    return features

class QLearningAgent(ReinforcementAgent):
    """
    A Q-Learning agent.

    Some functions that may be useful:

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getAlpha`:
    Get the learning rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getDiscountRate`:
    Get the discount rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`:
    Get the exploration probability.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getLegalActions`:
    Get the legal actions for a reinforcement agent.

    `pacai.util.probability.flipCoin`:
    Flip a coin (get a binary value) with some probability.

    `random.choice`:
    Pick randomly from a list.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Compute the action to take in the current state.
    With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
    we should take a random action and take the best policy action otherwise.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should choose None as the action.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    The parent class calls this to observe a state transition and reward.
    You should do your Q-Value update here.
    Note that you should never call this function, it will be called on your behalf.

    DESCRIPTION: <Write something here so we know what you did.>
    -Created update function to update q values from state, action, nextState, reward using the
    standard update function Q(s,a) <- Q(s,a) + alpha*[r + gamma*V(s') - Q(s,a)]
    
    -Fixed getQValue to return default value if q value not in dictionary

    -getPolicy return the action that yeilds the maximum expected value

    -getAction uses the epsilon to decide when to make a random choice
    and when to exploit the learned policy
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

        # You can initialize Q-values here.
        self.QValues = {}
    
    def update(self, state, action, nextState, reward):
        qValue = self.getQValue(state, action)
        alpha = self.getAlpha()
        gamma = self.getGamma()
        nextValue = self.getValue(nextState)
        # Q(s,a) <- Q(s,a) + alpha*[r + gamma*V(s') - Q(s,a)]
        self.QValues[(state, action)] = qValue + alpha * (reward + gamma * nextValue - qValue)

    def getQValue(self, state: AbstractGameState, action):
        """
        Get the Q-Value for a `pacai.core.gamestate.AbstractGameState`
        and `pacai.core.directions.Directions`.
        Should return 0.0 if the (state, action) pair has never been seen.
        """
        return self.QValues.get((state, action), 0.0)

    def getValue(self, state: AbstractGameState):
        """
        Return the value of the best action in a state.
        I.E., the value of the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of 0.0.

        This method pairs with `QLearningAgent.getPolicy`,
        which returns the actual best action.
        Whereas this method returns the value of the best action.
        """
        actions = state.getLegalActions(self.index)
        if not actions:
            return 0.0
        action_values = [self.getQValue(state, action) for action in actions]
        return max(action_values)

    def getPolicy(self, state: CaptureGameState):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of None.

        This method pairs with `QLearningAgent.getValue`,
        which returns the value of the best action.
        Whereas this method returns the best action itself.
        """
        # actions = self.getLegalActions(state)
        actions = state.getLegalActions(self.index)
        if actions is None:
            return None
        action_values = [self.getQValue(state, action) for action in actions]
        max_value = max(action_values)
        max_actions = [action
                       for action, value in zip(actions, action_values)
                       if value == max_value]

        return choice(max_actions)
    
    def getAction(self, state: CaptureGameState):
        if flipCoin(self.getEpsilon()):
            random_action = choice(state.getLegalActions(self.index))
            return random_action
        else:
            policy_action = self.getPolicy(state)
            return policy_action

class PacmanQAgent(QLearningAgent):
    """
    Exactly the same as `QLearningAgent`, but with different default parameters.
    """

    def __init__(self, index, epsilon = 0.05, gamma = 0.8, alpha = 0.2, numTraining = 0, **kwargs):
        kwargs['epsilon'] = epsilon
        kwargs['gamma'] = gamma
        kwargs['alpha'] = alpha
        kwargs['numTraining'] = numTraining

        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Simply calls the super getAction method and then informs the parent of an action for Pacman.
        Do not change or remove this method.
        """

        action = super().getAction(state)
        self.doAction(state, action)

        return action

class CaptureQAgent(PacmanQAgent, CaptureAgent):
    def __init__(self, index, weights=None, epsilon=0.5, gamma=0.75, alpha=0.0002,
                 numTraining=0, update_frequency=100, feature_extractor=betterFeatures,
                 debug=False):
        CaptureAgent.__init__(self, index)
        PacmanQAgent.__init__(
            self, index,
            epsilon=epsilon,
            gamma=gamma,
            alpha=alpha,
            numTraining=numTraining
        )

        self.startDiscountRate = self.getDiscountRate()
        self.startEpsilon = self.getEpsilon()
        self.startAlpha = self.getAlpha()

        # You might want to initialize weights here.
        # number of updates between setting target_weights to weights
        self.update_frequency = update_frequency
        self.getFeatures = feature_extractor        # function to get features
        self.debug = debug                          # outputs debug messages to console
        self.updates = 0                            # total number of updates stored
        self.weights = {}                           # weights updated every transition
        self.target_weights = {}                    # weights updates infrequently
        self.start_state = None                     # starting state of the current game

        if weights:
            self.weights = weights.copy()
            self.target_weights = weights.copy()

    def registerInitialState(self, gameState: CaptureGameState):
        """
        This method handles the initial setup of the agent and populates useful fields,
        such as the team the agent is on and the `pacai.core.distanceCalculator.Distancer`.

        IMPORTANT: If this method runs for more than 15 seconds, your agent will time out.
        """
        # get distances from board (maze distance)

        PacmanQAgent.registerInitialState(self, gameState)
        CaptureAgent.registerInitialState(self, gameState)
        self.start_state = gameState
        epsilon_multiple = (0.01 ** (self.episodesSoFar / self.numTraining)
                            if self.episodesSoFar < self.numTraining else 0)
        alpha_multiple = (0.5 ** (self.episodesSoFar / self.numTraining)
                          if self.episodesSoFar < self.numTraining else 0)

        self.epsilon = self.startEpsilon * epsilon_multiple
        self.alpha = self.startAlpha * alpha_multiple

        if (self.numTraining > 0):
            print((
                f"Episodes = {self.episodesSoFar}, "
                f"Learning Rate = {self.alpha}, "
                f"Epsilon = {self.epsilon}, "
                f"Weights = {self.target_weights}"
            ))

        elif (self.index == 0):
            print(f"Episodes = {self.episodesSoFar}")
                  
    def final(self, state):
        """
        Called at the end of each game.
        """

        # Call the super-class final method.
        super().final(state)

        # Did we finish training?
        if self.episodesSoFar == self.numTraining:
            pass
        
    def get_weight(self, feature):
        return self.target_weights.get(feature, 0)

    def getQValue(self, state, action):
        features_dict: dict = self.getFeatures(self, state, action)

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
        features_dict: dict = self.getFeatures(self, state, action)

        if self.debug:
            print(features_dict)

        for feature, value in features_dict.items():
            self.weights[feature] = self.get_weight(feature) + alpha * correction * value

        self.updates += 1
        if self.updates % self.update_frequency == 0:
            self.target_weights = self.weights.copy()

    def chooseAction(self, gameState):
        action = super().getPolicy(gameState)
        return action