import numpy as np
from scipy.stats import norm
import networkx as nx

import matplotlib.pyplot as plt

class MaxFlowGraph:
    '''
    Python implementation of Dinic's Algorithm
    This code is contributed by rupasriachanta421.
    Geeks for Geeks - https://www.geeksforgeeks.org/dinics-algorithm-maximum-flow/
    '''   
    class Edge:
        def __init__(self, v, flow, C, rev):
            self.v = v
            self.flow = flow
            self.C = C
            self.rev = rev

    def __init__(self, V):
        self.adj = [[] for i in range(V)]
        self.V = V
        self.level = [0 for i in range(V)]

    def addEdge(self, u, v, C):
        a = self.Edge(v, 0, C, len(self.adj[v]))
        b = self.Edge(u, 0, 0, len(self.adj[u]))
        self.adj[u].append(a)
        self.adj[v].append(b)

    def BFS(self, s, t):
        for i in range(self.V):
            self.level[i] = -1
        self.level[s] = 0
        q = []
        q.append(s)
        while q:
            u = q.pop(0)
            for i in range(len(self.adj[u])):
                e = self.adj[u][i]
                if self.level[e.v] < 0 and e.flow < e.C:
                    self.level[e.v] = self.level[u]+1
                    q.append(e.v)
        return False if self.level[t] < 0 else True

    def sendFlow(self, u, flow, t, start):
        if u == t:
            return flow
        while start[u] < len(self.adj[u]):
            e = self.adj[u][start[u]]
            if self.level[e.v] == self.level[u]+1 and e.flow < e.C:
                curr_flow = min(flow, e.C-e.flow)
                temp_flow = self.sendFlow(e.v, curr_flow, t, start)
                if temp_flow and temp_flow > 0:
                    e.flow += temp_flow
                    self.adj[e.v][e.rev].flow -= temp_flow
                    return temp_flow
            start[u] += 1

    def DinicMaxflow(self, s, t):
        if s == t:
            return -1
        total = 0
        while self.BFS(s, t) == True:
            start = [0 for i in range(self.V+1)]
            while True:
                flow = self.sendFlow(s, float('inf'), t, start)
                if not flow:
                    break
                total += flow
        return total

    def print_edges(self):
        '''
        Print the edges of the graph
        '''
        for u in range(self.V):
            for e in self.adj[u]:
                print(f"Edge from {u} to {e.v}, flow: {e.flow}, capacity: {e.C}")

    def plot_graph(self):
        '''
        Plot the graph using networkx and matplotlib
        '''
        G = nx.DiGraph()
        for u in range(self.V):
            for e in self.adj[u]:
                G.add_edge(u, e.v, flow=e.flow, capacity=e.C)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray', width=1.5, arrows=True)
        edge_labels = nx.get_edge_attributes(G, 'flow')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.show()


def compositions(players, actions):
    '''
    Returns all the permutations of players into sets of cardinality of the number of actions
    '''
    if actions == 1:
        return [[players]]
    else:
        pi = []
        for i in range(players+1):
            sub_permutations = compositions(players-i, actions-1)
            for sub_permutation in sub_permutations:
                pi.append([i] + sub_permutation)
        return pi


