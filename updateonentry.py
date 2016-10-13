import numpy as np
from call1 import call1
from dijkstra import dijkstra


class updateonentry(object):
    blockstate_new1 = 0

    __init__(p, s, d, flow_type, min_rate, flownumber, userpriority, s_new, d_new,
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
        if self.s == 0:
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
