import numpy as np
from call1 import call1
from dijkstra import dijkstra


class updateonentry1(object):

    def __init__(self, p, s, d, flow_type, min_rate, flownumber, userpriority, s_new, d_new,
                 flow_type_new, min_rate_new, flownumber_new, userpriority_new, path_final, wt_matx,
                 wt_matx_real, wt_matx_real1, blockstate):
        self.p = p
        self.s = s
        self.d = d
        self.flow_type = flow_type
        self.min_rate = min_rate
        self.flownumber = flownumber
        self.userpriority = userpriority
        self.s_new = s_new
        self.d_new = d_new
        self.flow_type_new = flow_type_new
        self.min_rate_new = min_rate_new
        self.flownumber_new = flownumber_new
        self.userpriority_new = userpriority_new
        self.path_final = path_final
        self.wt_matx = wt_matx
        self.wt_matx_real = wt_matx_real
        self.wt_matx_real1 = wt_matx_real1
        self.blockstate = blockstate

    def execute(self):
        [s1, s2] = np.shape(self.path_final)
        if self.flow_type_new == 0:
            # Calls call1.py
            cd = call1(self.p, self.s_new, self.d_new, self.flow_type_new, self.min_rate, self.wt_matx,
                       self.wt_matx_real, self.wt_matx_real1)
            cd.execute()
            path1 = cd.path
            wt_matx = cd.wt_matx
            wt_matx_real = cd.wt_matx_real
            wt_matx_real1 = cd.wt_matx_real1
            if path1 == np.zeros((p)):
                blockstate_new = 0  # Represents blockstate
            else:
                # s_new and d_new interchanged. Maybe for destination to source calculation
                cd = call1(self.p, self.d_new, self.s_new, self.flow_type_new, self.min_rate, self.wt_matx,
                           self.wt_matx_real, self.wt_matx_real1)
                cd.execute()
                path2 = cd.path
                wt_matx = cd.wt_matx
                wt_matx_real = cd.wt_matx_real
                wt_matx_real1 = cd.wt_matx_real1
                if path2 == np.zeros((self.p)):
                    blockstate_new = 0
                    # Call2
                else:
                    blockstate_new = 1
                    noofpaths = 1
                    for loop in range(0, s1, 1):
                        # Not sure about this loop[0] or loop[1]
                        if self.path_final[loop][0] == 0:
                            self.path_final[loop, :] = np.append(self.flownumber_new, self.flow_type_new,
                                                                 self.userpriority_new, noofpaths,
                                                                 self.min_rate_new, path1)
                            self.path_final[loop+1, :] = np.append(self.flownumber_new, self.flow_type_new,
                                                                   self.userpriority_new, noofpaths,
                                                                   self.min_rate_new, path2)
                            break
        elif self.flow_type_new == 1:
            cd = call1(self.p, self.s_new, self.d_new, self.flow_type_new, self.min_rate, self.wt_matx,
                       self.wt_matx_real, self.wt_matx_real1)
            cd.execute()
            path1 = cd.path
            wt_matx = cd.wt_matx
            wt_matx_real = cd.wt_matx_real
            wt_matx_real1 = cd.wt_matx_real1
            if path1 == np.zeros((p)):
                blockstate_new = 0
            else:
                blockstate_new = 1
                noofpaths = 1
                for loop in range(0, s1, 1):
                    # Not sure about this loop[0] or loop[1]
                    if self.path_final[loop][0] == 0:
                        self.path_final[loop, :] = np.append(self.flownumber_new, self.flow_type_new,
                                                             self.userpriority_new, noofpaths,
                                                             self.min_rate_new, path1)
        elif self.flow_type_new == 2:
            cd = call1(self.p, self.s_new, self.d_new, self.flow_type_new, self.min_rate, self.wt_matx,
                       self.wt_matx_real, self.wt_matx_real1)
            cd.execute()
            path1 = cd.path
            wt_matx = cd.wt_matx
            wt_matx_real = cd.wt_matx_real
            wt_matx_real1 = cd.wt_matx_real1
            if path1 == np.zeros((p)):
                blockstate_new = 0
            else:
                blockstate_new = 1
                noofpaths = 1
                for loop in range(0, s1, 1):
                    # Not sure about this loop[0] or loop[1]
                    if self.path_final[loop][0] == 0:
                        self.path_final[loop, :] = np.append(self.flownumber_new, self.flow_type_new,
                                                             self.userpriority_new, noofpaths,
                                                             self.min_rate_new, path1)
        else:
            dij = dijkstra(self.s_new, self.d_new, self.wt_matx)
            path1 = dij.execute()
            [path1size1, path1size] = np.shape(path1)
            if path1size < self.p:
                for loop in range(path1size, p, 1):
                    path1[loop] = 0
            blockstate_new = 1
            noofpaths = 1
            for loop in range(0, s1, 1):
                # Not sure about this loop[0] or loop[1]
                if self.path_final[loop][0] == 0:
                    self.path_final[loop, :] = np.append(self.flownumber_new, self.flow_type_new,
                                                         self.userpriority_new, noofpaths,
                                                         self.min_rate_new, path1)
        if blockstate_new == 1:
            s = np.append(self.s, self.s_new)
            d = np.append(self.d, self.d_new)
            flow_type = np.append(self.flow_type, self.flow_type_new)
            min_rate = np.append(self.min_rate, self.min_rate_new)
            flownumber = np.append(self.flownumber, self.flownumber_new)
            userpriority = np.append(self.userpriority, self.userpriority_new)
            blockstate = np.append(self.blockstate, self.blockstate_new)
