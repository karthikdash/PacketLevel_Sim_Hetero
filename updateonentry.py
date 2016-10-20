import numpy as np
from call1 import call1
from dijkstra import dijkstra
from blockreal import blockreal
from blocknonreal import blocknonreal


class updateonentry(object):
    blockstate_new1 = 0

    __init__(self, p, s, d, flow_type, min_rate, flownumber, userpriority, s_new, d_new,
             flow_type_new, min_rate_new, flownumber_new, userpriority_new, path_final,
             wt_matx, wt_matx_real, wt_matx_real1, eff_capacity_matx, blockstate):
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
        self.eff_capacity_matx = eff_capacity_matx
        self.blockstate = blockstate

    def execute(self):
        [s1, s2] = np.shape(self.path_final)
        self.blockstate_new1 = 1
        if self.flow_type_new == 0:
            wt_matx_initial = self.wt_matx
            wt_matx_real_initial = self.wt_matx_real
            wt_matx_real1_initial = self.wt_matx_real1
            path_final_intial = self.path_final
            # Calls call1.py
            cd = call1(self.p, self.s_new, self.d_new, self.flow_type_new, self.min_rate, self.wt_matx,
                       self.wt_matx_real, self.wt_matx_real1)
            cd.execute()
            path1 = cd.path
            self.wt_matx = cd.wt_matx
            self.wt_matx_real = cd.wt_matx_real
            self.wt_matx_real1 = cd.wt_matx_real1
            v = path1 == np.zeros((self.p))
            if v.all():
                self.blockstate_new1 = 0
                # BlockReal
                br = blockreal(self.s_new, self.d_new, self.min_rate_new, self.path_final, self.wt_matx_real, self.wt_matx_real1, self.wt_matx)
                self.pathreal = br.pathreal
                self.path_final = br.path_final
                self.wt_matx = br.wt_matx
                self.wt_matx_real = br.wt_matx_real
                self.wt_matx_real1 = br.wt_matx_real1

                path1 = self.pathreal
            v = path1 == np.zeros((self.p))
            if v.all():
                blockstate_new = 0  # Represents block statement
            else:
                # Calls call1.py
                # For destination to source
                cd = call1(self.p, self.d_new, self.s_new, self.flow_type_new, self.min_rate, self.wt_matx,
                           self.wt_matx_real, self.wt_matx_real1)
                cd.execute()
                path2 = cd.path
                self.wt_matx = cd.wt_matx
                self.wt_matx_real = cd.wt_matx_real
                self.wt_matx_real1 = cd.wt_matx_real1
                v = path2 == np.zeros((self.p))
                if v.all():
                    self.blockstate_new1 = 0
                    # BlockReal
                    # Source to Destination
                    br = blockreal(self.d_new, self.s_new, self.min_rate_new, self.path_final, self.wt_matx_real, self.wt_matx_real1, self.wt_matx)
                    self.pathreal = br.pathreal
                    self.path_final = br.path_final
                    self.wt_matx = br.wt_matx
                    self.wt_matx_real = br.wt_matx_real
                    self.wt_matx_real1 = br.wt_matx_real1
                    path2 = self.pathreal
                v = path2 == np.zeros((self.p))
                if v.all():
                    blockstate_new = 0
                    self.wt_matx = wt_matx_initial
                    self.wt_matx_real = wt_matx_real_initial
                    self.wt_matx_real1 = wt_matx_real1_initial
                    self.path_final = path_final_intial
                else:
                    blockstate_new = 1
                    noofpaths = 1
                    for loop in range(0, s1, 1):
                        if self.path_final[loop, 1] == 0:
                            v = [self.flownumber_new, self.flow_type_new,
                                 self.userpriority_new, noofpaths,
                                 self.min_rate_new]
                            self.path_final[loop, :] = np.concatenate((v, path1))
                            self.path_final[loop + 1, :] = np.concatenate((v, path2))
                            break
        elif
