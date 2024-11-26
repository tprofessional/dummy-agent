from pacai.util import reflection
from pacai.student.defensiveAgents import *

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

    firstAgent = DefenseAgentDQN
    secondAgent = DefenseAgentDQN

    print(firstIndex, secondIndex)

    return [
        firstAgent(firstIndex),
        secondAgent(secondIndex),
    ]
