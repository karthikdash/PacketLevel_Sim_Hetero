import numpy as np
import dijkstra from dijkstra
import random


class blockreal(object):
    __init__(self, s_new, d_new, min_rate_new, path_final, wt_matx_real, wt_matx_real1, wt_matx):
        self.s_new = s_new
        self.d_new = d_new
        self.min_rate_new = min_rate_new
        self.path_final = path_final
        self.wt_matx_real = wt_matx_real
        self.wt_matx_real1 = self.wt_matx_real1
        self.wt_matx = wt_matx

    def execute(object):
        [p1, p] = np.shape(self.wt_matx)
        with np.errstate(divide='ignore', invalid='ignore'):
            dummy1 = np.divide(1, self.wt_matx_real1)
            dummy2 = np.subtract(1, self.min_rate_new)
            dummy3 = dummy2 > 0
            dummy4 = np.divide(1, self.wt_matx)
            dummy5 = np.multiply(dummy4, dummy3)
            dummy6 = np.subtract(dummy5, self.min_rate_new)
            dummy7 = dummy6 > 0
            dummy8 = np.multiply(dummy6, dummy7)
            dummy9 = np.divide(1, dummy8)
        for o in range(0, self.p, 1):
            for j in range(0, self.p, 1):
                if dummy9[o][j] == -float('inf'):
                    dummy9[o][j] == float('inf')
        # dijkstra3 is not required. dijkstra is enough with more global variables
        dij = dijkstra(self.s, self.d, dummy4)
        path = dij.execute()
        cost = dij.cost
        visited = dij.visited
        parent = dij.parent
        # End of dijkstra
        paths = np.zeros((self.p - 1, self.p + 1))
        count = 1
        for loop in range(0, p, 1):
            if parent[loop] != 0:  # If there is a path
                t = loop
                path = loop
                while t != self.s_new:
                    par = parent[t]
                    path = np.append(par, path)
                    t = par
                [pathsize1, pathsize] = np.shape(path)
                if pathsize < p:
                    # Not sure pathsize + 1 or pathsize
                    for j in range(pathsize, self.p, 1):
                        path[j] = 0
                paths[count, :] = np.append(loop, path)
                count = count + 1
        Anodes2 = []
        for loop in range(0, p, 1):
            if parent[loop] != 0:
                Anodes2 = np.append(Anodes2, loop)
        [Anodes2size1, Anodes2size] = np.shape(Anodes2)
        Anodes1 = Anodes2
        # Might be count
        for loop in range(0, count - 1, 1):
            noofnodes = np.sum(paths[loop, 2:self.p+1] > 0)
            if noofnodes > 2:
                for loop1 in range(1, noofnodes - 1, 1):
                    node = paths[loop, loop1]
                    index = np.where(Anode2 == node)
                    Anodes1[index] = 0
        Anodes = []
        for loop in range(0, Anodes2size, 1):
            if Anodes1[loop] != 0:
                Anodes = np.append(Anodes, Anodes2[loop])
        # End of forming set a Nodes
        Anodes3 = Anodes
        [Anodes3size1, Anodes3size] = np.shape(Anodes3)
        [mPF, nPF] = np.shape(self.path_final)
        check = 0
        while Anodes3size > 0:
            index = random.randit(1, Anodes3size)
        # For Destination Node First
        if dummy3[Anodes3[index], self.d_new] > 0:
            # Routing Real-time flow
            for loop1 in range(0, p-1, 1):
                if paths[loop1, 1] == Anodes3[index]
