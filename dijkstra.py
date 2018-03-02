import numpy as np


class dijkstra(object):
    cost = []
    visited = []
    parent = []

    def __init__(self, s, d, wt_matx, min_rate):
        self.s = np.array(s, dtype=np.int)
        self.d = np.array(d, dtype=np.int)
        self.wt_matx = wt_matx
        self.min_rate = min_rate

    def execute(self):
        # print (self.wt_matx), "lordvader"
        # path = []
        # k11 = np.genfromtxt('K8.csv', delimiter=',', dtype=int)
        # [m1, n1] = np.shape(k11)
        # for i in range(0, m1):
        #     if k11[i][0] == self.s:
        #         for j in range(0, n1):
        #             if k11[i][j] == self.d:
        #                 path = k11[i]
        #                 break
        #
        # for k in range(1, n1-1):
        #     if path[k] != 0:
        #         if self.wt_matx[path[k-1]-1][path[k]-1] <= 0:
        #             return []
        # path = filter(lambda a: a != 0, path)
        # return path
        # if path[-1] == 0:
        #     return path[:-1]
        # return path


        [m, n] = np.shape(self.wt_matx)
        visited = np.zeros((n))
        parent = np.zeros((n))
        with np.errstate(divide='ignore', invalid='ignore'):
            cost = np.divide(1, np.zeros((n)))
        cost[self.s-1] = 0
        for y in range(0, n, 1):
            temp = []
            for h in range(0, n, 1):
                if visited[h] == 0:
                    temp = np.append(temp, cost[h])
                else:
                    temp = np.append(temp, float('inf'))
            nxt1 = temp.min()  # Minimum Value
            nxt = temp.argmin()  # Index of the Minimum Value
            # print nxt
            if nxt+1 == self.d:
                break
            if nxt1 == float('inf'):
                break
            visited[nxt] = 1
            for z in range(0, n, 1):
                if visited[z] == 0:
                    newcost = cost[nxt] + self.wt_matx[nxt, z]
                    if newcost < cost[z]:
                        parent[z] = nxt+1
                        cost[z] = newcost
        path = []
        if parent[self.d-1] != 0:
            t = self.d
            path = self.d
            while (t != self.s):
                p = int(parent[t-1])
                path = np.append(p, path)
                t = p
        # print path
        return path
        self.cost = cost
        self.visited = visited
        self.parent = parent
