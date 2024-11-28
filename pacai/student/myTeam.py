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

    weights20 = {'bias': -0.0001011266355530183,
                'min-capsule-distance': -0.0009538489318344521,
                'nearest-food': -0.002213939271498827,
                'number-of-food': -0.0014782898091452608,
                'closest-op-to-spawn': -0.0005415827436528108,
                'dist-from-spawn': -0.000677846260008125,
                'near-prey': -0.001961097627818202,
                'kills-prey': 1.7991642824025714e-05,
                'near-preditor': -0.0005430379088875296,
                'preditor-1-away': -0.00011515519687259614}

    weights57 = {'bias': -8.54439202241422e-05,
                 'min-capsule-distance': -0.001593006487503221,
                 'nearest-food': -0.0034689364346597597,
                 'number-of-food': -0.00135582191834622,
                 'closest-op-to-spawn': -5.2113102169178676e-05,
                 'dist-from-spawn': 0.00047392165693618674,
                 'near-prey': -0.002954485670500278,
                 'kills-prey': 1.8360569736113712e-05,
                 'near-preditor': 0.00034759392100765533,
                 'preditor-1-away': -0.00015194424802607786}

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

    # firstAgent = DefenseAgentDQN(firstIndex, 
    #   weights=weights65, numTraining=200, alpha=0.00001, epsilon=0, update_frequency=2000)
    firstAgent = DefenseAgentDQN(firstIndex, 
                                 weights=weights65, numTraining=0, alpha=0, epsilon=0)
    secondAgent = DefenseAgentDQN(secondIndex, 
                                  weights=weights64, numTraining=0, alpha=0, epsilon=0)

    return [firstAgent, secondAgent]
