"""
This file contains incomplete versions of some agents that can be selected to control Pacman.
You will complete their implementations.

Good luck and happy searching!
"""

import logging

from pacai.core.actions import Actions
from pacai.core.directions import Directions
from pacai.core.distance import manhattan
# from pacai.core.search import heuristic
from pacai.core.search.position import PositionSearchProblem
from pacai.core.search.problem import SearchProblem
from pacai.agents.base import BaseAgent
from pacai.agents.search.base import SearchAgent

class CornersProblem(SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function.
    See the `pacai.core.search.position.PositionSearchProblem` class for an example of
    a working SearchProblem.

    Additional methods to implement:

    `pacai.core.search.problem.SearchProblem.startingState`:
    Returns the start state (in your search space,
    NOT a `pacai.core.gamestate.AbstractGameState`).

    `pacai.core.search.problem.SearchProblem.isGoal`:
    Returns whether this search state is a goal state of the problem.

    `pacai.core.search.problem.SearchProblem.successorStates`:
    Returns successor states, the actions they require, and a cost of 1.
    The following code snippet may prove useful:
    ```
        successors = []

        for action in Directions.CARDINAL:
            x, y = currentPosition
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]

            if (not hitsWall):
                # Construct the successor.

        return successors
    ```
    """

    def __init__(self, startingGameState):
        super().__init__()

        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top = self.walls.getHeight() - 2
        right = self.walls.getWidth() - 2

        self.corners = ((1, 1), (1, top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                logging.warning('Warning: no food in corner ' + str(corner))

        self.startState = (self.startingPosition, frozenset(self.corners))

    def actionsCost(self, actions):
        """
        Returns the cost of a particular sequence of actions.
        If those actions include an illegal move, return 999999.
        This is implemented for you.
        """

        if (actions is None):
            return 999999

        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999

        return len(actions)
    
    def startingState(self):
        return self.startState

    def isGoal(self, state):
        if state[1]:    # false if remaining goals is empty set
            return False
        
        # Bookkeeping for display purposes (the highlight in the GUI).
        if (state[0] not in self._visitedLocations):
            self._visitedLocations.add(state[0])
            self._visitHistory.append(state[0])

        return True  # true when all goals are obtained and removed from state[1]

    def successorStates(self, state):
        successors = []

        for action in Directions.CARDINAL:
            pos, goals = state
            dx, dy = Actions.directionToVector(action)
            new_pos = (int(pos[0] + dx), int(pos[1] + dy))
            hitsWall = self.walls[new_pos[0]][new_pos[1]]
            if (not hitsWall):
                new_goals = frozenset(_ for _ in goals if _ != new_pos)  # removes goals reached
                new_state = (new_pos, new_goals)
                successor = (new_state, action, 1)
                successors.append(successor)

        # Bookkeeping for display purposes (the highlight in the GUI).
        self._numExpanded += 1
        if (state[0] not in self._visitedLocations):
            self._visitedLocations.add(state[0])
            self._visitHistory.append(state[0])

        return successors
    

def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

    This function should always return a number that is a lower bound
    on the shortest path from the state to a goal of the problem;
    i.e. it should be admissible.
    (You need not worry about consistency for this heuristic to receive full credit.)
    """

    def compute_dist_from_food_0(food_0):
        from pacai.util.queue import Queue
        fringe = Queue()
        G = {food_0: 0}

        def get_successors(cell):
            successors = []
            for dir in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                new_cell = (cell[0] + dir[0], cell[1] + dir[1])
                if not problem.walls[new_cell[0]][new_cell[1]]:
                    successors.append(new_cell)
            return successors

        def update(cell):
            cost = G[cell]
            for succ in get_successors(cell):
                if succ not in G:
                    G[succ] = cost + 1
                    fringe.push(succ)
        
        update(food_0)
        while not fringe.isEmpty():
            update(fringe.pop())
        
        return G
    
    if not hasattr(problem, 'heuristicInfo'):
        problem.heuristicInfo = {}

    if 'dist_from_food_0' not in problem.heuristicInfo:
        print(len(problem.walls.asList(False)))
        problem.heuristicInfo['dist_from_food_0'] = compute_dist_from_food_0(list(state[1])[0])
    
    def compute_dist(a, b):     # computes minimum distance in a more robust manner
        m_dist = manhattan(a, b)
        f0_dist = abs(problem.heuristicInfo['dist_from_food_0'][a]
                    - problem.heuristicInfo['dist_from_food_0'][b])
        return max(m_dist, f0_dist)

    # finds the distances between any 2 points including the current point
    # then sorts to return the sum of the (# remaining goals) first elements
    # this is gaurenteed to be admissable since the sum of the min distances is
    # the minimum possible distance for the number of remaining edges
    # may have issues with consistancy when multiple goals minimum distance is
    # to the current state's location but I think it is ok
    def sum_n_mins(start, goals):
        if not goals:
            return 0
        points = list(goals)
        points.append(start)
        distances = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                distances.append(compute_dist(points[i], points[j]))
        distances.sort()
        min_dists = sum(distances[:len(goals)])
        max_dist = max(distances)
        return max(min_dists, max_dist)

    return sum_n_mins(*state)    # 509 nodes expanded

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.
    First, try to come up with an admissible heuristic;
    almost all admissible heuristics will be consistent as well.

    If using A* ever finds a solution that is worse than what uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!
    On the other hand, inadmissible or inconsistent heuristics may find optimal solutions,
    so be careful.

    The state is a tuple (pacmanPosition, foodGrid) where foodGrid is a
    `pacai.core.grid.Grid` of either True or False.
    You can call `foodGrid.asList()` to get a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the problem.
    For example, `problem.walls` gives you a Grid of where the walls are.

    If you want to *store* information to be reused in other calls to the heuristic,
    there is a dictionary called problem.heuristicInfo that you can use.
    For example, if you only want to count the walls once and store that value, try:
    ```
    problem.heuristicInfo['wallCount'] = problem.walls.count()
    ```
    Subsequent calls to this heuristic can access problem.heuristicInfo['wallCount'].
    """

    def compute_dist_from_food_0(food_0):
        from pacai.util.queue import Queue
        fringe = Queue()
        G = {food_0: 0}

        def get_successors(cell):
            successors = []
            for dir in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                new_cell = (cell[0] + dir[0], cell[1] + dir[1])
                if not problem.walls[new_cell[0]][new_cell[1]]:
                    successors.append(new_cell)
            return successors

        def update(cell):
            cost = G[cell]
            for succ in get_successors(cell):
                if succ not in G:
                    G[succ] = cost + 1
                    fringe.push(succ)
        
        update(food_0)
        while not fringe.isEmpty():
            update(fringe.pop())
        
        return G

    if 'dist_from_food_0' not in problem.heuristicInfo:
        problem.heuristicInfo['dist_from_food_0'] = compute_dist_from_food_0(state[1].asList()[0])
    
    def compute_dist(a, b):     # computes minimum distance in a more robust manner
        m_dist = manhattan(a, b)
        f0_dist = abs(problem.heuristicInfo['dist_from_food_0'][a]
                    - problem.heuristicInfo['dist_from_food_0'][b])
        return max(m_dist, f0_dist)

    pos = state[0]

    # sum of the n shortest edges where n is the number of remaining goals
    def minimum_pairwise_sum(pos, goals):
        if not goals:
            return 0
        points = list(goals)
        points.append(pos)
        distances = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                distances.append(compute_dist(points[i], points[j]))
        distances.sort()
        min_distances = sum(distances[:len(goals)])
        max_distance = max(distances)
        return max(0 * min_distances, max_distance)
    
    return minimum_pairwise_sum(pos, state[1].asList())



    def farthest_goal(pos, goals):
        if not goals:
            return 0
        return max([compute_dist(pos, goal) for goal in goals])
    
    def mean_distance(pos, goals):
        if not goals:
            return 0
        return sum([compute_dist(pos, goal) for goal in goals]) / len(goals)
    
    def remaining_goals(pos, goals):    # not valid Heuristic
        return 5 * len(goals)
    
    def bigSearch(pos, goals, adjacent_costs = [15, 5, 5, 2, 0]):  # not valid Heuristic
        def generate_neighbor_food():
            problem.heuristicInfo['neighbor_food'] = {}
            for goal in goals:
                update_neighbor_food(goal)

        def update_neighbor_food(goal):
            food_adjacent = 0
            for dir in ((0, 1), (0, -1), (1, 0), (-1, 0)):  # Up, Down, Right, Left
                adjacent = (goal[0] + dir[0], goal[1] + dir[1])
                if adjacent in goals:
                    food_adjacent += 1
            problem.heuristicInfo['neighbor_food'][goal] = adjacent_costs[food_adjacent]

        def update_neighbors(goal):
            for dir in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                neighbor = (goal[0] + dir[0], goal[1] + dir[1])
                if neighbor in goals:
                    update_neighbor_food(neighbor)

        if 'neighbor_food' not in problem.heuristicInfo:
            generate_neighbor_food()

        def get_cost(goal):
            if goal not in problem.heuristicInfo['neighbor_food']:  # Not sure how this even
                update_neighbor_food(goal)                          # happens but this is necessary
            return min(problem.heuristicInfo['neighbor_food'][goal], len(goals))

        update_neighbors(pos)
        return sum([get_cost(goal) for goal in goals])

    adjacent_costs = [15, 5, 5, 2, 0]       # 314 with 517 nodes
    adjacent_costs = [15, 5, 5, 1, -2]      # 312 with 647 nodes
    adjacent_costs = [15, 1, 1, 0, 0]       # 309 with 22057 nodes
    adjacent_costs = [14, 1, 1, 0, -1]      # 307 with 16651 nodes
    adjacent_costs = [13, 1, 1, 0, -1]      # 305 with 27550 nodes
    adjacent_costs = [999, 1, 1, 0, -1]     # 284 with 5132 nodes
    adjacent_costs = [64, 1, 1, 0, -0.1]    # 284 with 3448 nodes

    return bigSearch(pos, state[1].asList(), adjacent_costs=adjacent_costs)

class ClosestDotSearchAgent(SearchAgent):
    """
    Search for all food using a sequence of searches.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def registerInitialState(self, state):
        self._actions = []
        self._actionIndex = 0

        currentState = state

        while (currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState)  # The missing piece
            self._actions += nextPathSegment

            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' %
                            (str(action), str(currentState)))

                currentState = currentState.generateSuccessor(0, action)

        logging.info('Path found with cost %d.' % len(self._actions))

    def findPathToClosestDot(self, gameState):
        from pacai.student.search import breadthFirstSearch
        problem = AnyFoodSearchProblem(gameState)
        return breadthFirstSearch(problem)

class AnyFoodSearchProblem(PositionSearchProblem):

    def __init__(self, gameState, start = None):
        super().__init__(gameState, goal = None, start = start)

        self.walls = gameState.getWalls()
        self.startingPosition = gameState.getPacmanPosition()
        self.food = frozenset(gameState.getFood().asList())
        self.startState = (self.startingPosition, self.food)

    def actionsCost(self, actions):
        """
        Returns the cost of a particular sequence of actions.
        If those actions include an illegal move, return 999999.
        This is implemented for you.
        """

        if (actions is None):
            return 999999

        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999

        return len(actions)
    
    def startingState(self):
        return self.startState

    def isGoal(self, state):
        if state[0] not in self.startState[1]:    # false if position not in food
            return False
        
        # Bookkeeping for display purposes (the highlight in the GUI).
        if (state[0] not in self._visitedLocations):
            self._visitedLocations.add(state[0])
            self._visitHistory.append(state[0])

        return True  # true when all goals are obtained and removed from state[2]

    def successorStates(self, state):
        successors = []

        for action in Directions.CARDINAL:
            pos, food = state
            dxy = Actions.directionToVector(action)
            new_pos = (int(pos[0] + dxy[0]), int(pos[1] + dxy[1]))
            if (not self.walls[new_pos[0]][new_pos[1]]):
                new_food = frozenset(_ for _ in food if _ != new_pos)  # removes goals reached
                new_state = (new_pos, new_food)
                successor = (new_state, action, 1)
                successors.append(successor)

        # Bookkeeping for display purposes (the highlight in the GUI).
        self._numExpanded += 1
        if (state[0] not in self._visitedLocations):
            self._visitedLocations.add(state[0])
            self._visitHistory.append(state[0])

        return successors


from pacai.core.search.food import FoodSearchProblem
from pacai.student.search import aStarSearch

class ApproximateSearchAgent(BaseAgent):
    """
    Implement your contest entry here.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Get a `pacai.bin.pacman.PacmanGameState`
    and return a `pacai.core.directions.Directions`.

    `pacai.agents.base.BaseAgent.registerInitialState`:
    This method is called before any moves are made.
    """

    

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        self.moves = []
        self.move_number = 0

    def getAction(self, state):
        """
        The BaseAgent will receive an `pacai.core.gamestate.AbstractGameState`,
        and must return an action from `pacai.core.directions.Directions`.
        """
        from pacai.core.directions import Directions
        if self.move_number < len(self.moves):
            move = self.moves[self.move_number]
            self.move_number += 1
            return move
        return Directions.STOP

    def registerInitialState(self, state):

        """
        Inspect the starting state.
        """
        self.gameState = state

        
        def adjacent_count(goal, goals, dirs=((0, 1), (0, -1), (1, 0), (-1, 0))):
            count = 0
            for dx, dy in dirs:
                count += (goal[0] + dx, goal[1] + dy) in goals
            return count
        

        self.problem = FoodSearchProblem(state)
        self.goals = set(state.getFood().asList())

        self.visited = []





        self.moves = aStarSearch(FoodSearchProblem(state), self.bigSearch)

        self.graphSimp(self.goals)
        # self.gameState.setHighlightLocations(self.moves)
        # self.moves = self.simplifiedSearch((15, 1), self.graph)




    def graphSimp(self, goals):
        def getAdjacent(goal, goals=goals, dirs=((0, 1), (0, -1), (1, 0), (-1, 0))):
            adjacent = []
            for dx, dy in dirs:
                adj = (goal[0] + dx, goal[1] + dy)
                if adj in goals:
                    adjacent.append(adj)
            return adjacent

        def get_dead(graph):
            dead = [goal for goal, adj in graph.items() if len(adj) == 1]
            return dead
        
        def add_path(node, path):
            node_paths.setdefault(node, []).extend(path)

        def get_adjacent_dir(node0, node1):  # nodes must be adjacent
            return Actions.vectorToDirection((node1[0] - node0[0], node1[1] - node0[1]))
        
        def get_adjacent_path(nodes):
            return [get_adjacent_dir(nodes[i], nodes[i+1]) for i in range(len(nodes) - 1)]
        
        def remove_from_graph(node, graph):
            for adj in graph[node]:
                graph[adj].remove(node)
            del graph[node]

        def trim_dead(graph):
            while True:
                dead_nodes = get_dead(graph)
                if not dead_nodes:
                    break
                self.visited += dead_nodes
                for node in dead_nodes:
                    superNode = graph[node][0]
                    path =  get_adjacent_path([superNode, node]) + \
                            node_paths[node] + \
                            get_adjacent_path([node, superNode])
                    add_path(superNode, path)
                    remove_from_graph(node, graph)
                    del node_paths[node]

        goals.add((15, 1))
        graph = {goal : getAdjacent(goal) for goal in goals}
        goals.remove((15, 1))
        node_paths = {goal : [] for goal in goals}     # stores the path for simplified nodes

        trim_dead(graph)

        self.gameState.setHighlightLocations(self.visited)

    def bigSearch(self, state, problem, adjacent_costs=[64, 1, 1, 0, -0.1]):
        pos = state[0]
        goals = state[1].asList()

        def generate_neighbor_food():
            problem.heuristicInfo['neighbor_food'] = {}
            for goal in goals:
                update_neighbor_food(goal)

        def update_neighbor_food(goal):
            food_adjacent = 0
            for dir in ((0, 1), (0, -1), (1, 0), (-1, 0)):  # Up, Down, Right, Left
                adjacent = (goal[0] + dir[0], goal[1] + dir[1])
                if adjacent in goals:
                    food_adjacent += 1
            problem.heuristicInfo['neighbor_food'][goal] = adjacent_costs[food_adjacent]

        def update_neighbors(goal):
            for dir in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                neighbor = (goal[0] + dir[0], goal[1] + dir[1])
                if neighbor in goals:
                    update_neighbor_food(neighbor)

        if 'neighbor_food' not in problem.heuristicInfo:
            generate_neighbor_food()

        def get_cost(goal):
            if goal not in problem.heuristicInfo['neighbor_food']:  # Not sure how this even
                update_neighbor_food(goal)                          # happens but this is necessary
            return min(problem.heuristicInfo['neighbor_food'][goal], len(goals))

        update_neighbors(pos)

        return sum([get_cost(goal) for goal in goals])


class DistanceProblem:
    def __init__(self, links, startingState):
        self.links = links
        self.startState = startingState

    def startingState(self):
        return self.startState

    def isGoal(self, state):
        print("State = ", state)
        return len(state[1]) == 0
    
    def successorStates(self, state):
        successors = []
        pos, goals = state
        for new_pos, _, cost in self.links[pos]:
            new_goals = frozenset([goal for goal in goals if goal != pos])
            new_state = (new_pos, new_goals)
            successors.append((new_state, new_pos, cost))
            print(new_pos)
        return successors







"""
        def has_shared_diagonal(goal, goals):
            shared_dirs = \
                (((0, 1), (1, 1), (1, 0)), 
                 ((1, 0), (1, -1), (0, -1)),
                 ((0, -1), (-1, -1), (-1, 0)),
                 ((-1, 0), (-1, 1), (0, 1)))
            for dirs in shared_dirs:
                if adjacent_count(goal, goals, dirs=dirs) == 3:
                    return True
            return False
        
        def get_adjacents(goal, goals, dirs=((0, 1), (0, -1), (1, 0), (-1, 0))):
            adjacents = []
            for dx, dy in dirs:
                adjacent = (goal[0] + dx, goal[1] + dy)
                if adjacent in goals:
                    adjacents.append(adjacent)
            return adjacents   
        
        def find_junctions(goals):
            junctions = [goal for goal in goals if 
                adjacent_count(goal, goals) > 2 and not has_shared_diagonal(goal, goals)]
            junctions.append((1, 13))   # break the loop in the top left
            junctions.append((28, 10))  # add goal at end of dead end in top right
            junctions.append((15, 1))   # add junction at start pos
            return set(junctions)
        
        # Each edge has a set of points, 2 junctions and 2 paths from junction to junction
        def make_graph(goals, junctions):
            def propagate_connected(node, nodes):
                from pacai.util.queue import Queue
                hall = set()
                fringe = Queue()
                fringe.push(node)
                while not fringe.isEmpty():
                    elem = fringe.pop()
                    hall.add(elem)
                    for adj in get_adjacents(elem, nodes):
                        if adj not in hall and adj not in fringe.list:
                            fringe.push(adj)
                return hall
            
            nodes = goals - junctions
            halls = []

            for node in nodes:
                done_nodes = {node for edge in halls for node in edge}
                if node not in done_nodes:
                    halls.append(propagate_connected(node, nodes))
            
            nodes_hall = {node : hall for hall in halls for node in hall}            
            junction_halls = {junction : [] for junction in junctions}  # halls adj. to junction
            hall_junctions = {frozenset(hall) : [] for hall in halls}   # junctions adj. to hall

            for junction in junctions:
                for node in get_adjacents(junction, goals):
                    hall = nodes_hall[node]
                    junction_halls[junction].append(hall)
                    hall_junctions[frozenset(hall)].append(junction)


            # dictionary of costs from junction to junction
            # cost is len(hall) + 1
            # path is array of moves
            # each links = {from: [(to, [moves], cost)]}
            from pacai.core.directions import Directions

            graph = {junction: [] for junction in junctions}
            for junction in junctions:
                for hall in junction_halls[junction]:
                    is_dead = len(hall_junctions[frozenset(hall)]) == 1
                    other_junction = junction
                    if not is_dead:
                        other_junction = [elem for elem in hall_junctions[frozenset(hall)] \
                            if elem != junction][0]
                    # solve mini all food problem starting at other junction
                    # the walls should be present and there should be no food at junction
                    # reverse the path order and directions to get path out of junction across edge\
                    from copy import deepcopy
                    miniState = deepcopy(state)
                    miniState._agentStates[0]._position = other_junction  # start search from other
                    for x, y in miniState.getFood().asList():
                        if (x, y) not in hall:
                            miniState.eatFood(x, y)
                    miniFoodProblem = FoodSearchProblem(miniState)
                    heur = lambda succ, _: len(succ[1].asList()) + 0.5*manhattan(junction, succ[0])
                    # use heuristic that returns towards junction
                    reverse_path = aStarSearch(miniFoodProblem, heur)

                    x, y = other_junction
                    for dir in reverse_path:
                        dx, dy = Actions._directions[dir]
                        x += dx
                        y += dy
                    startFoodPath = (x,  y)

                    foodPath = [Directions.REVERSE[move] for move in reverse_path]
                    foodPath.reverse()  # correct path going out of junction towards other_junction

                    print("FP", foodPath)

                    # find path from junction to end of food path
                    miniState = deepcopy(state)
                    miniState._agentStates[0]._position = other_junction
                    miniPathProblem = PositionSearchProblem(miniState, goal=startFoodPath)
                    heur = lambda succ, _: manhattan(startFoodPath, succ)
                    startPath = aStarSearch(miniPathProblem, heur)
                    path = startPath + foodPath
    
                    print("StartP", startPath)


                    cost = len(path)
                    edge = (other_junction, tuple(path), cost)
                    graph[junction].append(edge)
            return graph
"""