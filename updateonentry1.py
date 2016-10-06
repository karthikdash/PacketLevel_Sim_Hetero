import numpy as np
import call1 as call1

Class updateonentry1:
    def _init_(self, p, s, d, flow_type, min_rate, flownumber, userpriority, s_new, d_new,
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
                if path2 = zeros(1, self.p):
                    blockstate_new = 0
                    # Call2
                else:
                    blockstate_new = 1
                    noofpaths = 1
                    for loop in range(0, s1, 1):
                        # Not sure about this loop[0] or loop[1]
                        if self.path_final[loop][0] == 0:
                            self.path_final[loop, :] = np.append(self.flownumber_new, self.)
