import numpy as np
import dijkstra as dijkstra


class call2(object):
    wt_matx = []
    wt_matx_real = []
    wt_matx_real1 = []

    def __init__(self, p, path, flow_type, min_rate, wt_matx, wt_matx_real, wt_matx_real1):
        self.p = p
        self.path = path
        self.flow_type = flow_type
        self.min_rate = min_rate
        self.wt_matx = wt_matx
        self.wt_matx_real = wt_matx_real
        self.wt_matx_real1 = wt_matx_real1

    def execute(self):
        if self.flow_type == 2:
            with np.errstate(divide='ignore', invalid='ignore'):
                dummy = np.divide(1, self.wt_matx)
            s2 = np.shape(self.path)[0]
            flow = np.zeros((self.p, self.p))
            for i in range(0, s2-1, 1):
                if self.path[i + 1] == 0:
                    break
                else:
                    flow[int(self.path[i] - 1), int(self.path[i+1] - 1)] = self.min_rate
            dummy2 = np.add(dummy, flow)
            with np.errstate(divide='ignore', invalid='ignore'):
                self.wt_matx = np.divide(1, dummy2)
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                dummy = np.divide(1, self.wt_matx_real)
                dummy1 = np.divide(1, self.wt_matx_real1)
                dummy2 = np.divide(1, self.wt_matx)
            s2 = np.shape(self.path)[0]
            flow = np.zeros((self.p, self.p))
            for i in range(0, s2-1, 1):
                if self.path[i+1] == 0:
                    break
                else:
                    # print self.path[i], "selfish"
                    # print flow
                    flow[int(self.path[i]-1), int(self.path[i+1]-1)] = self.min_rate
            dummy3 = np.add(dummy, flow)
            dummy4 = np.add(dummy1, flow)
            dummy5 = np.add(dummy2, flow)
            with np.errstate(divide='ignore', invalid='ignore'):
                self.wt_matx_real = np.divide(1, dummy3)
                self.wt_matx_real1 = np.divide(1, dummy4)
                self.wt_matx = np.divide(1, dummy5)
