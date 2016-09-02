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
        self.epsilon = 0.8   # probability of randomly select action
        self.alpha = 0.3  # Q-value learning rate
        self.discount = 0.9  # discount for future rewards
        self.Qvalue = OrderedDict()  # initialize Q-value table
        self.previous_state = None # initialize previous state  
        self.rounds = 1  # initialize training rounds


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.rounds += 1 # increase count of training rounds 
        if self.rounds > 100:
            self.rounds = 100        

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        previous_state = self.env.agent_states[self]
        previous_location = previous_state['location']
        previous_heading = previous_state['heading']

        self.epsilon = self.epsilon * (100-self.rounds)/100  # decrease epsilon with increasing training rounds as confidence gains along

        # TODO: Update state with sensor information and guided directions towards destination

        self.previous_state = (inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'], self.next_waypoint)

        for a in self.env.valid_actions:  # initialize Q-value table for unknown combinations of states and actions
            if (self.previous_state, a) not in self.Qvalue.iterkeys():
                self.Qvalue[(self.previous_state, a)] = 0
           
        prob = np.array([self.Qvalue[(self.previous_state, None)] , self.Qvalue[(self.previous_state, 'forward')], self.Qvalue[(self.previous_state, 'left')], self.Qvalue[(self.previous_state, 'right')]])
        #prob = np.exp(prob)/np.sum(np.exp(prob), axis = 0)
        #print prob
        # TODO: Select action according to your policy

        if random.random() < self.epsilon:
            action = np.random.choice(self.env.valid_actions)#, p=prob)
        else:
            action = self.env.valid_actions[np.random.choice(np.where(prob == prob.max())[0])]  # prefer this rather than argmax, which can break the tie preference 
 

        # Execute action and get reward
        oldValue = self.Qvalue[(self.previous_state, action)]
        reward = self.env.act(self, action)
        
        # TODO: Learn policy based on state, action, reward

        state = self.env.agent_states[self]
        location = state['location']
        heading = state['heading']

        nextInputs = self.env.sense(self)
        #print inputs, nextInputs
        self.state = (nextInputs['light'], nextInputs['oncoming'], nextInputs['left'], nextInputs['right'], self.planner.next_waypoint())

        for a in self.env.valid_actions: # initialize Q-value table for unknown combinations of current states and actions
            if (self.state, a) not in self.Qvalue.iterkeys():
                self.Qvalue[(self.state, a)] = 0

        futureValue = max([self.Qvalue[(self.state, a)] for a in self.env.valid_actions])
        newValue = reward + self.discount * futureValue 
        self.Qvalue[(self.previous_state, action)] =  (1-self.alpha) * oldValue + self.alpha * newValue  # update Q-value table balanced by learning rate
        
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]A
        #print self.previous_state, previous_location, previous_heading, action, self.Qvalue[(self.previous_state, action)], location, heading, self.state, futureValue


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    for (state, action) in a.Qvalue.iterkeys(): 
        print (state, action), a.Qvalue[(state, action)]

    Simulator(e, update_delay = 1, display = True).run(n_trials = 5)

if __name__ == '__main__':
    run()
