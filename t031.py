from random import randint
from random import random
import math

# ---------------------------------------------------- #
#                    GRUPO 31                          #
# ---------------------------------------------------- #
# 78631 - Maria Sbrancia  &   90777 - Rodrigo Rosa     #
# ---------------------------------------------------- #



class LearningAgent:
        # init
        # nS maximum number of states
        # nA maximum number of action per state
        def __init__(self,nS,nA):

                # define this function
                self.nS = nS
                self.nA = nA
                self.alpha = 0.4
                self.gamma = 0.7
                self.e = 0.99
                self.fMax = 700
                self.Q = [[-math.inf for i in range(nA)] for k in range(nS)]
                self.N = [[0 for i in range(nA)] for k in range(nS)]
              
        
        # Select one action, used when learning  
        # st - is the current state        
        # aa - is the set of possible actions
        # for a given state they are always given in the same order
        # returns
        # a - the index to the action in aa
        def selectactiontolearn(self,st,aa):

                if (random() <= self.e):
                        min_ind = []
                        m = self.N[st][0]
                        for i in range(1, len(aa)):
                                if (self.N[st][i] < m):
                                        min_ind = [i]
                                        m = self.N[st][i]
                        for i in range(len(aa)):
                                if self.N[st][i] == m:
                                        min_ind.append(i)

                        if (self.e > 0.15):
                                self.e *= 0.993 

                        if (self.alpha > 0.18):
                                self.alpha *= 0.9993

                        if (self.gamma <= 0.9):
                                self.gamma *= 1.05

                        
                        if m > self.fMax:
                                return randint(0, len(aa)-1)
                        else:
                                return min_ind[randint(0, len(min_ind)-1)]
                     
                else:   
                        m = self.Q[st][0]
                        m_index = 0

                        for i in range(1, len(aa)):
                                if (self.Q[st][i] > m) and (self.N[st][i] < self.fMax):
                                        m = self.Q[st][i]
                                        m_index = i
                        return m_index

        # Select one action, used when evaluating
        # st - is the current state        
        # aa - is the set of possible actions
        # for a given state they are always given in the same order
        # returns
        # a - the index to the action in aa
        def selectactiontoexecute(self,st,aa):
                m = self.Q[st][0]
                m_index = 0
                for i in range(1, len(aa)):
                        if self.Q[st][i] > m:
                                m = self.Q[st][i]
                                m_index = i
                return m_index


        # this function is called after every action
        # st - original state
        # nst - next state
        # a - the index to the action taken
        # r - reward obtained
        def learn(self,ost,nst,a,r):
                self.N[ost][a] += 1

                maxNextQ = self.Q[nst][0]
                for i in range(1, self.nA):
                        maxNextQ = max(self.Q[nst][i], maxNextQ)
                        if (math.isinf(maxNextQ)):
                                maxNextQ = 0

                if (not math.isinf(self.Q[ost][a])):
                        self.Q[ost][a] = self.Q[ost][a] + self.alpha *(r + self.gamma * maxNextQ - self.Q[ost][a])

                else:
                        self.Q[ost][a] = r + self.gamma * maxNextQ
                


