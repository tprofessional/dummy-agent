from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.core.gamestate import AbstractGameState
from pacai.util import reflection

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
        actions = self.getLegalActions(state)
        if not actions:
            return 0.0
        action_values = [self.getQValue(state, action) for action in actions]
        return max(action_values)

    def getPolicy(self, state):
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
        actions = self.getLegalActions(state)
        if not actions:
            return None
        action_values = [self.getQValue(state, action) for action in actions]
        max_index = action_values.index(max(action_values))
        return actions[max_index]
    
    def getAction(self, state):
        from random import choice
        from pacai.util.probability import flipCoin
        if flipCoin(self.getEpsilon()):
            random_action = choice(self.getLegalActions(state))
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

class ApproximateQAgent(PacmanQAgent):
    """
    An approximate Q-learning agent.

    You should only have to overwrite `QLearningAgent.getQValue`
    and `pacai.agents.learning.reinforcement.ReinforcementAgent.update`.
    All other `QLearningAgent` functions should work as is.

    Additional methods to implement:

    `QLearningAgent.getQValue`:
    Should return `Q(state, action) = w * featureVector`,
    where `*` is the dotProduct operator.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    Should update your weights based on transition.

    DESCRIPTION: <Write something here so we know what you did.>
    Added weights which is a dictionary for each of the features extracted
    and computes the dot product between the weights and the values of the features.
    This is then used to update the weights and learn a function between the wegith values
    observed and the approximate q value for state action pairs.
    """

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

    def getQValue(self, state: AbstractGameState, action):
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
            