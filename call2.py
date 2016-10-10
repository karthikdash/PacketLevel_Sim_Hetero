import numpy as np
import dijkstra as dijkstra


class call2(object):
    wt_matx = []
    wt_matx_real = []
    wt_matx_real1 = []

    def _init_(self, p, path, flow_type, min_rate, wt_matx, wt_matx_real, wt_matx_real1):
        self.p = p
        self.path = path
        self.flow_type = flow_type
        self.min_rate = min_rate
        self.wt_matx = wt_matx
        self.wt_matx_real = wt_matx_real
        self.wt_matx_real1 = wt_matx_real1

    def execute(self):
        if self.flow_type == 2:
            dummy = np.divide(1, self.wt_matx)
            [s1, s2] = np.shape(self.path)
            flow = np.zeros((self.p))
            for i in range(0, s2-1, 1):
                if self.path[i + 1] == 0:
                    break
                else:
                    flow[path[i], path[i+1]] = self.min_rate
            dummy2 = np.add(dummy, flow)
            wt_matx = np.divide(1, dummy2)
        else:
            dummy = np.divide(1, self.x1=wt_matx_real)
            dummy1 = np.divide(1, self.wt_matx_real1)
            dummy2 = np.divide(1, self.wt_matx)
            [s1, s2] = np.shape(self.path)
            flows = np.zeros((self.p))
            for i in range(0, s2-1, 1):
                if path[i+1] == 0:
                    break
                else:
                    flow[self.path[i], self.path[i+1]] = self.min_rate
            dummy3 = np.add(dummy, flow)
            dummy4 = np.add(dummy1, flow)
            dummy5 = np.add(dummy2, flow)
            wt_matx_real = np.divide(1, dummy3)
            wt_matx_real1 = np.divide(1, dummy4)
            wt_matx = np.divide(1, dummy5)
