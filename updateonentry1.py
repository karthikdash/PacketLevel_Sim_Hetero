import numpy as np
from call1 import call1
from call2 import call2
from dijkstra import dijkstra


class updateonentry1(object):

    s = []
    d = []
    flow_type = []
    min_rate = []
    flownumber = []
    userpriority = []
    blockstate = []
    blockstate_new = []
    wt_matx = []
    wt_matx_real = []
    wt_matx_real1 = []

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
            # Source to destination calculation
            cd = call1(self.p, self.s_new, self.d_new, self.flow_type_new, self.min_rate_new, self.wt_matx,
                       self.wt_matx_real, self.wt_matx_real1)
            cd.execute()
            path1 = cd.path
            self.wt_matx = cd.wt_matx
            self.wt_matx_real = cd.wt_matx_real
            self.wt_matx_real1 = cd.wt_matx_real1
            np.savetxt("pathfinal_flowtype0.csv", path1, delimiter=",")
            v = path1 == np.zeros((self.p))
            if v.all():
                self.blockstate_new = 0  # Represents blockstate
            else:
                # Destination to source calculation
                cd = call1(self.p, self.d_new, self.s_new, self.flow_type_new, self.min_rate_new, self.wt_matx,
                           self.wt_matx_real, self.wt_matx_real1)
                cd.execute()
                path2 = cd.path
                self.wt_matx = cd.wt_matx
                self.wt_matx_real = cd.wt_matx_real
                self.wt_matx_real1 = cd.wt_matx_real1
                v = path2 == np.zeros((self.p))
                if v.all():
                    self.blockstate_new = 0
                    # Call2
                    # If blocked by destination to source, we invoke Call2
                    cd1 = call2(self.p, path1, self.flow_type_new, self.min_rate_new, self.wt_matx,
                                self.wt_matx_real, self.wt_matx_real1)
                    cd1.execute()
                    self.wt_matx = cd1.wt_matx
                    self.wt_matx_real = cd1.wt_matx_real
                    self.wt_matx_real1 = cd1.wt_matx_real1
                else:
                    self.blockstate_new = 1
                    noofpaths = 1
                    for loop in range(0, s1-1, 1):
                        # Not sure about this loop[0] or loop[1]
                        if self.path_final[loop][0] == 0:
                            # print self.path_final
                            # print path1
                            v = [self.flownumber_new, self.flow_type_new,
                                 self.userpriority_new, noofpaths,
                                 self.min_rate_new]
                            self.path_final[loop, :] = np.concatenate((v, path1))
                            v1 = [self.flownumber_new, self.flow_type_new,
                                  self.userpriority_new, noofpaths,
                                  self.min_rate_new]
                            self.path_final[loop+1, :] = np.concatenate((v1, path2))
                            np.savetxt("pathfinal1.csv", self.path_final, delimiter=",")
                            break
        elif self.flow_type_new == 1:
            cd = call1(self.p, self.s_new, self.d_new, self.flow_type_new, self.min_rate_new, self.wt_matx,
                       self.wt_matx_real, self.wt_matx_real1)
            cd.execute()
            path1 = cd.path
            self.wt_matx = cd.wt_matx
            self.wt_matx_real = cd.wt_matx_real
            self.wt_matx_real1 = cd.wt_matx_real1
            if path1 == np.zeros((p)):
                self.blockstate_new = 0
            else:
                self.blockstate_new = 1
                noofpaths = 1
                for loop in range(0, s1, 1):
                    # Not sure about this loop[0] or loop[1]
                    if self.path_final[loop][0] == 0:
                        self.path_final[loop, :] = np.append(self.flownumber_new, self.flow_type_new,
                                                             self.userpriority_new, noofpaths,
                                                             self.min_rate_new, path1)
                        np.savetxt("pathfinal2.csv", self.path_final, delimiter=",")
                        break
        elif self.flow_type_new == 2:
            cd = call1(self.p, self.s_new, self.d_new, self.flow_type_new, self.min_rate_new, self.wt_matx,
                       self.wt_matx_real, self.wt_matx_real1)
            cd.execute()
            path1 = cd.path
            # print "PATH12"
            # print path1
            self.wt_matx = cd.wt_matx
            self.wt_matx_real = cd.wt_matx_real
            self.wt_matx_real1 = cd.wt_matx_real1
            v = path1 == np.zeros((self.p))
            if v.all():
                self.blockstate_new = 0
            else:
                self.blockstate_new = 1
                noofpaths = 1
                for loop in range(0, s1, 1):
                    # print loop
                    # Not sure about this loop[0] or loop[1]
                    if self.path_final[loop][0] == 0:
                        v = [self.flownumber_new, self.flow_type_new,
                             self.userpriority_new, noofpaths,
                             self.min_rate_new]
                        self.path_final[loop, :] = np.concatenate((v, path1))
                        break
        else:
            dij = dijkstra(self.s_new, self.d_new, self.wt_matx)
            path1 = dij.execute()
            [path1size1, path1size] = np.shape(path1)
            if path1size < self.p:
                for loop in range(path1size, p, 1):
                    path1[loop] = 0
            self.blockstate_new = 1
            noofpaths = 1
            for loop in range(0, s1, 1):
                # Not sure about this loop[0] or loop[1]
                if self.path_final[loop][0] == 0:
                    v = [self.flownumber_new, self.flow_type_new,
                         self.userpriority_new, noofpaths,
                         self.min_rate_new]
                    self.path_final[loop, :] = np.concatenate((v, path1))
                    np.savetxt("pathfinal3.csv", self.path_final, delimiter=",")
                    break
        if self.blockstate_new == 1:
            self.s = np.append(self.s, self.s_new)
            self.d = np.append(self.d, self.d_new)
            self.flow_type = np.append(self.flow_type, self.flow_type_new)
            self.min_rate = np.append(self.min_rate, self.min_rate_new)
            self.flownumber = np.append(self.flownumber, self.flownumber_new)
            self.userpriority = np.append(self.userpriority, self.userpriority_new)
            self.blockstate = np.append(self.blockstate, self.blockstate_new)
