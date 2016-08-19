import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import OrderedDict
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.epsilon = 0.5
        self.alpha = 0.3
        self.discount = 0.9
        self.Qvalue = OrderedDict()
        self.state = None



    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        # self.Qvalue = OrderedDict()        

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        #self.epsilon = self.epsilon * deadline/100
        # TODO: Update state
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'], self.next_waypoint)

        for action in self.env.valid_actions:
            if (self.state, action) not in self.Qvalue.iterkeys():
                self.Qvalue[(self.state, action)] = 0
           
        prob = [self.Qvalue[(self.state, None)] , self.Qvalue[(self.state, 'forward')], self.Qvalue[(self.state, 'left')], self.Qvalue[(self.state, 'right')]]
        prob = np.exp(prob)/np.sum(np.exp(prob), axis = 0)

        # TODO: Select action according to your policy

        if random.random() < self.epsilon:
            action = np.random.choice(self.env.valid_actions, p=prob)
        else:
            action = self.env.valid_actions[np.argmax(prob)]
 

        # Execute action and get reward
        oldValue = self.Qvalue[(self.state, action)]
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward

        #nextInputs = self.env.sense(self)
        #print inputs, nextInputs
        #nextState = (nextInputs['light'], nextInputs['oncoming'], nextInputs['left'], nextInputs['right'], self.planner.next_waypoint())

        #for action in self.env.valid_actions:
        #    if (nextState, action) not in self.Qvalue.iterkeys():
        #        self.Qvalue[(nextState, action)] = 0

        #futureValue = max([self.Qvalue[(nextState, a)] for a in self.env.valid_actions])
        newValue = reward #+ self.discount * futureValue 
        self.Qvalue[(self.state, action)] =  (1-self.alpha) * oldValue + self.alpha * newValue
        
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]A
        #print self.state, action, self.Qvalue[(self.state, action)], nextState, futureValue


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.01, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    for (state, action) in a.Qvalue.iterkeys(): 
        print (state, action), a.Qvalue[(state, action)]

    Simulator(e, update_delay = 1, display = True).run(n_trials = 5)

if __name__ == '__main__':
    run()
