import numpy as np


class dijkstra(object):

    def __init__(self, s, d, wt_matx):
        self.s = s
        self.d = d
        self.wt_matx = wt_matx

    def execute(self):
        print (self.wt_matx)
        [m, n] = np.shape(self.wt_matx)
        visited = np.zeros((n))
        parent = np.zeros((n))
        with np.errstate(divide='ignore', invalid='ignore'):
            cost = np.divide(1, np.zeros((n)))
        cost[self.s] = 0
        for y in range(0, n, 1):
            temp = []
            for h in range(0, n, 1):
                if visited[h] == 0:
                    temp = np.append(temp, cost[h])
                else:
                    temp = np.append(temp, float('inf'))
            nxt1 = temp.min()  # Minimum Value
            nxt = temp.argmin()  # Index of the Minimum Value
            print nxt
            if nxt == self.d:
                break
            if nxt1 == float('inf'):
                break
            visited[nxt] = 1
            for z in range(0, n, 1):
                if visited[z] == 0:
                    newcost = cost[nxt] + self.wt_matx[nxt, z]
                    if newcost < cost[z]:
                        parent[z] = nxt
                        cost[z] = newcost
        path = []
        if parent[self.d] != 0:
            t = self.d
            path = self.d
            while (t != s):
                p = parent(t)
                path = np.append(p, path)
                t = p
        return path
