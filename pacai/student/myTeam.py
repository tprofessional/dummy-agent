from pacai.agents.capture.capture import CaptureAgent
from pacai.bin.capture import CaptureGameState
from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.bin.capture import CaptureGameState
from pacai.util import util

import random
from random import random, choice

import time

def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.student.defensiveAgents.DummyAgent1',
        second = 'pacai.student.defensiveAgents.DummyAgent1'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    # firstAgent = reflection.qualifiedImport(first)
    # secondAgent = reflection.qualifiedImport(second)

    # 63%
    # weights = {'bias': -8.921822791015062e-05,
    #            'min-capsule-distance': -0.004895693381128324,
    #            'nearest-food': -0.003921470464572105,
    #            'number-of-food': -0.0023570021059283047,
    #            'closest-op-to-spawn': 0.0007079519210047101,
    #            'near-prey': -0.005176871054061819,
    #            'near-preditor': 0.000818904174007504}

    # 64%
    weights64 = {'bias': -8.855406482894479e-05,
                'min-capsule-distance': -0.004969176113354762,
                'nearest-food': -0.004061433122826846,
                'number-of-food': -0.002340619632591361,
                'closest-op-to-spawn': 0.0007394920809822344,
                'near-prey': -0.005268531370921854,
                'near-preditor': 0.0008454392419300111,
                'dist-from-spawn': 9.77400528029471e-05,
                'preditor-1-away': -3.197302339773383e-06,
                'kills-prey': 3.6307053467143223e-06}

    # weights20 = {'bias': -0.0001011266355530183,
    #             'min-capsule-distance': -0.0009538489318344521,
    #             'nearest-food': -0.002213939271498827,
    #             'number-of-food': -0.0014782898091452608,
    #             'closest-op-to-spawn': -0.0005415827436528108,
    #             'dist-from-spawn': -0.000677846260008125,
    #             'near-prey': -0.001961097627818202,
    #             'kills-prey': 1.7991642824025714e-05,
    #             'near-preditor': -0.0005430379088875296,
    #             'preditor-1-away': -0.00011515519687259614}

    # weights57 = {'bias': -8.54439202241422e-05,
    #              'min-capsule-distance': -0.001593006487503221,
    #              'nearest-food': -0.0034689364346597597,
    #              'number-of-food': -0.00135582191834622,
    #              'closest-op-to-spawn': -5.2113102169178676e-05,
    #              'dist-from-spawn': 0.00047392165693618674,
    #              'near-prey': -0.002954485670500278,
    #              'kills-prey': 1.8360569736113712e-05,
    #              'near-preditor': 0.00034759392100765533,
    #              'preditor-1-away': -0.00015194424802607786}

    weights65 = {'bias': -8.629683049651184e-05,
                 'min-capsule-distance': -0.0016011666836693072,
                 'nearest-food': -0.003491814117702272,
                 'number-of-food': -0.0013725675397402376,
                 'closest-op-to-spawn': -5.289008986728529e-05,
                 'dist-from-spawn': 0.0004703318508064342,
                 'near-prey': -0.0029634303592519047,
                 'kills-prey': 1.836418199981366e-05,
                 'near-preditor': 0.0003266220620258886,
                 'preditor-1-away': -0.0001535800627560156}

    firstAgent = DefenseAgentDQN(
        firstIndex,
        weights=weights65,
        numTraining=0,
        alpha=0,
        epsilon=0
    )
    
    secondAgent = DefenseAgentDQN(
        secondIndex,
        weights=weights64,
        numTraining=0,
        alpha=0,
        epsilon=0
    )

    return [firstAgent, secondAgent]

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
    next_self_state: AgentState = nextGameState.getAgentState(self.index)
    team_states: list[AgentState] = [gameState.getAgentState(idx) for idx in self.getTeam(gameState)]
    ops_states: list[AgentState] = [gameState.getAgentState(idx) for idx in self.getOpponents(gameState)]
    ops_next_states: list[AgentState] = [nextGameState.getAgentState(idx) for idx in self.getOpponents(nextGameState)]

    ourFood = self.getFoodYouAreDefending(gameState)
    opsFood = self.getFood(gameState)

    ourCapsules = self.getCapsulesYouAreDefending(gameState)
    opsCapsules = self.getCapsules(gameState)

    def is_prey(op: AgentState):
        return (next_self_state.isBraveGhost() and op.isPacman()) or (next_self_state.isPacman() and op.isScaredGhost())
    
    def is_preditor(op: AgentState):
        return (next_self_state.isPacman() and op.isBraveGhost()) or (next_self_state.isScaredGhost() and op.isPacman())
    
    to_ints = lambda x: (int(x[0]), int(x[1]))
    vec_add = lambda x, y: (x[0] + y[0], x[1] + y[1])
    safe_min = lambda arr: min(arr) if arr else 0

    pos = to_ints(gameState.getAgentPosition(self.index))
    next_pos = to_ints(nextGameState.getAgentPosition(self.index))
    start_pos = to_ints(self.start_state.getAgentPosition(self.index))
    teamate_pos = to_ints(gameState.getAgentPosition((self.index + 2) % 4))

    prey_dists = sorted([maze(next_pos, to_ints(op.getPosition())) for op in ops_states if is_prey(op)])
    preditor_dists = sorted([maze(next_pos, to_ints(op.getPosition())) for op in ops_states if is_preditor(op)])
    dists_from_spawn = sorted([abs(start_pos[0] - op.getPosition()[0]) for op in ops_next_states])

    if len(prey_dists) > 0:
        features['near-prey'] = prey_dists[0]
        features['kills-prey'] = prey_dists[0] == 0

    # if len(prey_dists) > 1:
    #     features['far-prey'] = prey_dists[1]

    if len(preditor_dists) > 0:
        features['near-preditor'] = preditor_dists[0]
        features['preditor-1-away'] = preditor_dists[0] == 1
        # features['dies'] = maze(pos, next_pos) > 1    # fix this

    # if len(preditor_dists) > 1:
    #     features['far-preditor'] = preditor_dists[1]

    if len(opsCapsules) != 0:
        min_dist = safe_min([maze(next_pos, cap) for cap in opsCapsules])
        features['min-capsule-distance'] = min_dist

    if len(opsFood.asList()) != 0:
        min_dist = safe_min([maze(next_pos, food) for food in opsFood.asList()])
        features['nearest-food'] = min_dist

    # features['dist-from-start'] = min(10, maze(next_pos, start_pos))

    # features['dist-from-teamate'] = maze(next_pos, teamate_pos)

    features['number-of-food'] = len(self.getFood(nextGameState).asList()) - 2

    # closest op to our spawn along the x-axis
    features['closest-op-to-spawn'] = safe_min(dists_from_spawn)

    # distance from spawn along the x-axis
    features['dist-from-spawn'] = abs(next_pos[0] - start_pos[0])

    # if self.red:
    #     features['score'] = nextGameState.getScore()
    # else:
    #     features['score'] = -nextGameState.getScore()

    for key in features:
        features[key] /= 10.0

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

    from pacai.core.gamestate import AbstractGameState

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
        max_actions = [action for action, value in zip(actions, action_values) if value == max_value]

        return choice(max_actions)
    
    def getAction(self, state: CaptureGameState):
        from random import choice
        from pacai.util.probability import flipCoin
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

class DefenseAgentDQN(PacmanQAgent, CaptureAgent):
    def __init__(self, index, weights=None, epsilon=0.5, gamma=0.75, alpha=0.0002, numTraining=0, update_frequency=100):
        CaptureAgent.__init__(self, index)
        PacmanQAgent.__init__(self, index, epsilon=epsilon, gamma=gamma, alpha=alpha, numTraining=numTraining)

        self.startDiscountRate = self.getDiscountRate()
        self.startEpsilon = self.getEpsilon()
        self.startAlpha = self.getAlpha()

        # You might want to initialize weights here.
        self.update_frequency = update_frequency    # number of updates between setting target_weights to weights
        self.updates = 0                            # total number of updates stored
        self.weights = {}                           # weights updated every transition
        self.target_weights = {}                    # weights updates infrequently
        self.start_state = None                     # starting state of the current game

        if weights:
            self.weights = weights.copy()
            self.target_weights = weights.copy()

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
        epsilon_multiple = 0.01 ** (self.episodesSoFar / self.numTraining) if self.episodesSoFar < self.numTraining else 0
        alpha_multiple = 0.5 ** (self.episodesSoFar / self.numTraining) if self.episodesSoFar < self.numTraining else 0

        self.epsilon = self.startEpsilon * epsilon_multiple
        self.alpha = self.startAlpha * alpha_multiple

        if (self.numTraining > 0):
            print(f"Episodes = {self.episodesSoFar}, Learning Rate = {self.alpha}, Epsilon = {self.epsilon}, Weghts = {self.target_weights}")
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
            # You might want to print your weights here for debugging.
            # *** Your Code Here ***
            # print(self.weights)
            pass
        
    def get_weight(self, feature):
        return self.target_weights.get(feature, 0)

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

        self.updates += 1
        if self.updates % self.update_frequency == 0:
            self.target_weights = self.weights.copy()

    def chooseAction(self, gameState):
        action = super().getPolicy(gameState)
        return action

# Copied over from pacai/agents/capture/dummy.py 
# class DummyAgent1(CaptureAgent):
#     """
#     A Dummy agent to serve as an example of the necessary agent structure.
#     You should look at `pacai.core.baselineTeam` for more details about how to create an agent.
#     """

#     def __init__(self, index, **kwargs):
#         super().__init__(index, **kwargs)

#     def chooseAction(self, gameState):
#         """
#         Picks among the actions with the highest return from `ReflexCaptureAgent.evaluate`.
#         """

#         actions = gameState.getLegalActions(self.index)

#         start = time.time()
#         values = [self.evaluate(gameState, a) for a in actions]
#         logging.debug('evaluate() time for agent %d: %.4f' % (self.index, time.time() - start))

#         maxValue = max(values)
#         bestActions = [a for a, v in zip(actions, values) if v == maxValue]

#         return random.choice(bestActions)

#     def getSuccessor(self, gameState, action):
#         """
#         Finds the next successor which is a grid position (location tuple).
#         """

#         successor = gameState.generateSuccessor(self.index, action)
#         pos = successor.getAgentState(self.index).getPosition()

#         if (pos != util.nearestPoint(pos)):
#             # Only half a grid position was covered.
#             return successor.generateSuccessor(self.index, action)
#         else:
#             return successor

#     def evaluate(self, gameState, action):
#         """
#         Computes a linear combination of features and feature weights.
#         """

#         features = self.getFeatures(gameState, action)
#         weights = self.getWeights(gameState, action)
#         stateEval = sum(features[feature] * weights[feature] for feature in features)

#         return stateEval

#     def getFeatures(self, gameState, action):
#         features = {}
#         successor = self.getSuccessor(gameState, action)
#         features['successorScore'] = self.getScore(successor)

#         # Compute distance to the nearest food.
#         foodList = self.getFood(successor).asList()

#         # This should always be True, but better safe than sorry.
#         if (len(foodList) > 0):
#             myPos = successor.getAgentState(self.index).getPosition()
#             minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
#             features['distanceToFood'] = minDistance

#         return features

#     def getWeights(self, gameState, action):
#         return {
#             'successorScore': 100,
#             'distanceToFood': -1
#         }
