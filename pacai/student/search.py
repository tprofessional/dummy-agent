"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""
from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue


def solveSearch(problem, is_goal, fringe, *extra):
    visited = set()
    pred = {}

    # generate the arguments tuple to be unpacked into is_goal function
    arguments = (visited, pred, fringe, *extra) if extra else (visited, pred, fringe, *extra)

    # initialize the fringe by checking if the starting state is the goal
    goal = is_goal(problem.startingState(), *arguments)

    # pop state from the fringe and update the fringe with new states that can be reached
    while not fringe.isEmpty():
            current_item = fringe.pop()
            goal = is_goal(current_item, *arguments)
            if goal:
                break


    # reverse the path starting at the goal and backtracking to start
    solution = []
    if goal:
        curr = goal
        while curr in pred:
            prev, action = pred[curr]
            solution.append(action)
            curr = prev
    solution.reverse()

    return solution
    

####################################################################################################

def depthFirstSearch(problem):

    def is_goal(state, visited, pred, fringe):
        visited.add(state)
        if problem.isGoal(state):
            return state
        for succ, action, cost in problem.successorStates(state):
            if succ in visited:
                continue
            fringe.push(succ)
            pred[succ] = (state, action)
        return None

    return solveSearch(problem, is_goal, Stack())
    
####################################################################################################

def breadthFirstSearch(problem):
        
    def is_goal(state, visited, pred, fringe):
        visited.add(state)
        if problem.isGoal(state):
            return state
        for succ, action, cost in problem.successorStates(state):
            if succ in visited or succ in fringe.list:  # BFS must check if succ in fringe
                continue
            fringe.push(succ)
            pred[succ] = (state, action)
        return None

    return solveSearch(problem, is_goal, Queue())


####################################################################################################

def uniformCostSearch(problem):

    def is_goal(state, visited, pred, fringe, G):
        visited.add(state)
        if problem.isGoal(state):
            return state
        for succ, action, cost in problem.successorStates(state):
            succ_dist = G[succ] if succ in G else float('inf')
            new_dist = G[state] + cost
            if (new_dist < succ_dist):
                G[succ] = new_dist
                fringe.push(succ, G[succ])
                pred[succ] = (state, action)
        return None

    G = {problem.startingState(): 0}   # minimum distance from start to node

    return solveSearch(problem, is_goal, PriorityQueue(), G)

####################################################################################################

def aStarSearch(problem, heuristic):

    def is_goal(state, visited, pred, fringe, G):
        visited.add(state)
        if problem.isGoal(state):
            return state
        for succ, action, cost in problem.successorStates(state):
            succ_dist = G[succ] if succ in G else float('inf')
            new_dist = G[state] + cost
            if (new_dist < succ_dist):
                G[succ] = new_dist
                heur_cost = G[succ] + heuristic(succ, problem)
                fringe.push(succ, heur_cost)
                pred[succ] = (state, action)
        return None

    G = {problem.startingState(): 0}   # minimum distance from start to node

    return solveSearch(problem, is_goal, PriorityQueue(), G)
