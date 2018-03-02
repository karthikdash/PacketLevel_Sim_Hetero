import numpy as np
from dijkstra import dijkstra


class call1(object):
    path = []
    wt_matx = []
    wt_matx_real = []
    wt_matx_real1 = []

    def __init__(self, p, s, d, flow_type, min_rate, wt_matx, wt_matx_real, wt_matx_real1, connection_type):
        self.p = p
        self.s = s
        self.d = d
        self.flow_type = flow_type
        self.min_rate = min_rate
        self.wt_matx = wt_matx
        self.wt_matx_real = wt_matx_real
        self.wt_matx_real1 = wt_matx_real1
        self.connection_type = connection_type


    def execute(self):
        if self.flow_type == 2:
            with np.errstate(divide='ignore', invalid='ignore'):
                dummy = np.divide(1, self.wt_matx)
                dummy1 = np.subtract(dummy, self.min_rate)
                # dummy2 = dummy1 > 0
                dummy1[dummy1 < 0] = 0
                dummy1[dummy1 > 0] = 1
                dummy2 = dummy1
                dummy1 = np.subtract(dummy, self.min_rate)
                dummy3 = np.multiply(dummy1, dummy2)
                dummy4 = np.divide(1, dummy3)
            for i in range(0, self.p, 1):
                for j in range(0, self.p, 1):
                    # Could be an erroneous declaration of negative Inf
                    if np.isneginf(dummy4[i][j]):
                        dummy4[i][j] = float('inf')
            # dij = dijkstra(self.s, self.d, dummy3, self.min_rate)
            dij = dijkstra(self.s, self.d, dummy4, self.min_rate)
            self.path = dij.execute()
            s2 = np.shape(self.path)[0]
            if s2 == 0:
                self.path = np.zeros((self.p))
            else:
                flow = np.zeros((self.p, self.p))
                for i in range(0, s2-1, 1):
                    # print self.min_rate, "selfminrate"
                    flow[self.path[i]-1, self.path[i+1]-1] = self.min_rate
                dummy5 = dummy-flow
                with np.errstate(divide='ignore', invalid='ignore'):
                    self.wt_matx = np.divide(1, dummy5)
                if s2 < self.p:
                    for j in range(s2, self.p, 1):
                        self.path = np.append(self.path, 0)
        else:

            dummy1 = np.divide(1, self.wt_matx_real1)
            # print self.min_rate, "selfminrar"
            dummy2 = np.subtract(dummy1, self.min_rate)
            # dummy3 = dummy2 > 0
            dummy2[dummy2 < 0] = 0
            dummy2[dummy2 > 0] = 1
            dummy3 = dummy2
            with np.errstate(divide='ignore', invalid='ignore'):
                dummy4 = np.divide(1, self.wt_matx)
            dummy5 = np.multiply(dummy4, dummy3)
            dummy6 = np.subtract(dummy5, self.min_rate)
            dummy6[dummy6 < 0] = 0
            dummy6[dummy6 > 0] = 1
            dummy7 = np.subtract(dummy5, self.min_rate)
            dummy8 = np.multiply(dummy6, dummy7)
            with np.errstate(divide='ignore', invalid='ignore'):
                dummy9 = np.divide(1, dummy8)
            for i in range(0, self.p, 1):
                for j in range(0, self.p, 1):
                    # print (dummy9[i][j])
                    if np.isneginf(dummy9[i][j]):
                        dummy9[i][j] = float('inf')

            # dij = dijkstra(self.s, self.d, dummy8, self.min_rate)
            dij = dijkstra(self.s, self.d, dummy9, self.min_rate)
            self.path = dij.execute()
            s2 = np.shape(self.path)[0]
            if s2 == 0:
                self.path = np.zeros((self.p))
            else:
                flow = np.zeros((self.p, self.p))
                # print flow
                for i in range(0, s2 - 1, 1):
                    flow[self.path[i]-1, self.path[i+1]-1] = self.min_rate
                dummy10 = np.subtract(np.divide(1, self.wt_matx_real), flow)
                # Toggle this to separate voice and video thresholds
                dummy1 = np.divide(1, self.wt_matx_real1)
                dummy11 = np.subtract(dummy1, flow)
                dummy12 = np.subtract(dummy4, flow)
                with np.errstate(divide='ignore', invalid='ignore'):
                    self.wt_matx_real = np.divide(1, dummy10)
                    self.wt_matx_real1 = np.divide(1, dummy11)
                    self.wt_matx = np.divide(1, dummy12)
                if s2 < self.p:
                    for j in range(s2, self.p, 1):
                        self.path = np.append(self.path, 0)
                # print self.path
