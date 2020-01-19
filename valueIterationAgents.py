# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for iteration in range(0, self.iterations):
            states = self.mdp.getStates()
            previousValues = self.values.copy()
            for state in states:
                possibleActions = self.mdp.getPossibleActions(state)
                # stateValueSums = dictionary ---> Key: Action Value: Action values
                stateValueSums = {}
                for possibleAction in possibleActions:
                    stateProbPairs = self.mdp.getTransitionStatesAndProbs(state, possibleAction)
                    currentStateValueSum = 0
                    for stateProb in stateProbPairs:
                        # Update
                        currentStateValueSum = currentStateValueSum + (stateProb[1] * (self.mdp.getReward(state, possibleAction, stateProb[0]) + self.discount * previousValues[stateProb[0]]))
                    stateValueSums[possibleAction] = currentStateValueSum
                if stateValueSums:
                    self.values[state] = max(list(stateValueSums.values()))

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        sum = 0
        previousValues = self.values
        stateProbPairs = self.mdp.getTransitionStatesAndProbs(state, action)
        for stateProb in stateProbPairs:
            # Update
            sum = sum + (stateProb[1] * (self.mdp.getReward(state, action, stateProb[0]) + self.discount * previousValues[stateProb[0]]))
        
        return sum
        
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        values = self.values
        if self.mdp.isTerminal(state) or (not self.mdp.getPossibleActions(state)):
            return None
        else:
            possibleActions = self.mdp.getPossibleActions(state)
            maxSoFar = -1000000
            maxActionSoFar = ""
            for possibleAction in possibleActions:
                currentActionValue = self.computeQValueFromValues(state, possibleAction)
                if currentActionValue > maxSoFar:
                    maxSoFar = currentActionValue
                    maxActionSoFar = possibleAction
                    
            return maxActionSoFar

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for iteration in range(0, self.iterations):
            state = states[iteration % len(states)]
            previousValues = self.values.copy()
            possibleActions = self.mdp.getPossibleActions(state)
            # stateValueSums = dictionary ---> Key: Action Value: Action values
            stateValueSums = {}
            for possibleAction in possibleActions:
                stateProbPairs = self.mdp.getTransitionStatesAndProbs(state, possibleAction)
                currentStateValueSum = 0
                for stateProb in stateProbPairs:
                    # Update
                    currentStateValueSum = currentStateValueSum + (stateProb[1] * (self.mdp.getReward(state, possibleAction, stateProb[0]) + self.discount * previousValues[stateProb[0]]))
                stateValueSums[possibleAction] = currentStateValueSum
            if stateValueSums:
                self.values[state] = max(list(stateValueSums.values()))

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Compute predecessors of all states
        initialValues = self.values.copy()
        states = self.mdp.getStates()
        statesWithPredStates = {}
        predStates = set()
        for state in states:
            possibleActions = self.mdp.getPossibleActions(state)
            for possibleAction in possibleActions:
                stateProbPairs = self.mdp.getTransitionStatesAndProbs(state, possibleAction)
                for stateProb in stateProbPairs:
                    nextState = stateProb[0]
                    predStates.add(state)
                    statesWithPredStates[nextState] = predStates
        
        # Initialize an empty priority queue
        pq = util.PriorityQueue()
        for state in states:
            if not self.mdp.isTerminal(state):
                # For all non-terminal states do...
                actionOfState = ValueIterationAgent.computeActionFromValues(self, state)
                diff = abs(initialValues[state] - ValueIterationAgent.computeQValueFromValues(self, state, actionOfState))
                pq.push(state, -diff)
                
        for iteration in range(0, self.iterations):
            if pq.isEmpty():
                return
            s = pq.pop()
            if not self.mdp.isTerminal(s):
                a = self.computeActionFromValues(s)
                sValue = self.computeQValueFromValues(s, a)
                self.values[s] = sValue
                predecessorsOfS = statesWithPredStates[s]
                for predecessorOfS in predecessorsOfS:
                    actionOfS = self.computeActionFromValues(predecessorOfS)
                    diff = abs(self.values[predecessorOfS] - self.computeQValueFromValues(predecessorOfS, actionOfS))
                    if diff > self.theta:
                        pq.update(predecessorOfS, -diff)
        
        
        
        
        
        
        
        
        
        
        
        
        
        

