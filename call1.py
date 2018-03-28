import numpy as np
from dijkstra import dijkstra


def call1(p, s, d, flow_type, min_rate, wt_matx, wt_matx_real, wt_matx_real1):
    if flow_type == 2:
        with np.errstate(divide='ignore', invalid='ignore'):
            dummy = np.divide(1, wt_matx)
            dummy1 = np.subtract(dummy, min_rate)
            # dummy2 = dummy1 > 0
            dummy1[dummy1 < 0] = 0
            dummy1[dummy1 > 0] = 1
            dummy2 = dummy1
            dummy1 = np.subtract(dummy, min_rate)
            dummy3 = np.multiply(dummy1, dummy2)
            dummy4 = np.divide(1, dummy3)
        for i in range(0, p, 1):
            for j in range(0, p, 1):
                # Could be an erroneous declaration of negative Inf
                if np.isneginf(dummy4[i][j]):
                    dummy4[i][j] = float('inf')
        # dij = dijkstra(s, d, dummy3, min_rate)
        dij = dijkstra(s, d, dummy4, min_rate)
        path = dij.execute()
        s2 = np.shape(path)[0]
        if s2 == 0:
            path = np.zeros((p))
        else:
            flow = np.zeros((p, p))
            for i in range(0, s2-1, 1):
                # print min_rate, "selfminrate"
                flow[path[i]-1, path[i+1]-1] = min_rate
            dummy5 = dummy-flow
            with np.errstate(divide='ignore', invalid='ignore'):
                wt_matx = np.divide(1, dummy5)
            if s2 < p:
                for j in range(s2, p, 1):
                    path = np.append(path, 0)
    else:

        dummy1 = np.divide(1, wt_matx_real1)
        # print min_rate, "selfminrar"
        dummy2 = np.subtract(dummy1, min_rate)
        # dummy3 = dummy2 > 0
        dummy2[dummy2 < 0] = 0
        dummy2[dummy2 > 0] = 1
        dummy3 = dummy2
        with np.errstate(divide='ignore', invalid='ignore'):
            dummy4 = np.divide(1, wt_matx)
        dummy5 = np.multiply(dummy4, dummy3)
        dummy6 = np.subtract(dummy5, min_rate)
        dummy6[dummy6 < 0] = 0
        dummy6[dummy6 > 0] = 1
        dummy7 = np.subtract(dummy5, min_rate)
        dummy8 = np.multiply(dummy6, dummy7)
        with np.errstate(divide='ignore', invalid='ignore'):
            dummy9 = np.divide(1, dummy8)
        for i in range(0, p, 1):
            for j in range(0, p, 1):
                # print (dummy9[i][j])
                if np.isneginf(dummy9[i][j]):
                    dummy9[i][j] = float('inf')

        # dij = dijkstra(s, d, dummy8, min_rate)
        dij = dijkstra(s, d, dummy9, min_rate)
        path = dij.execute()
        s2 = np.shape(path)[0]
        if s2 == 0:
            path = np.zeros((p))
        else:
            flow = np.zeros((p, p))
            # print flow
            for i in range(0, s2 - 1, 1):
                flow[path[i]-1, path[i+1]-1] = min_rate
            dummy10 = np.subtract(np.divide(1, wt_matx_real), flow)
            # Toggle this to separate voice and video thresholds
            dummy1 = np.divide(1, wt_matx_real1)
            dummy11 = np.subtract(dummy1, flow)
            dummy12 = np.subtract(dummy4, flow)
            with np.errstate(divide='ignore', invalid='ignore'):
                wt_matx_real = np.divide(1, dummy10)
                wt_matx_real1 = np.divide(1, dummy11)
                wt_matx = np.divide(1, dummy12)
            if s2 < p:
                for j in range(s2, p, 1):
                    path = np.append(path, 0)
            # print path

    return wt_matx, wt_matx_real, wt_matx_real1, path
