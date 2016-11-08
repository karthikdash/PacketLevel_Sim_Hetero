import numpy as np

class allocaterealupdated1(object):

    def __init__(self, p, s_multi, d_multi, flow_type_multi, min_rate_multi, flownumber_multi, userpriority_multi, s_new, d_new, flow_type_new,
                 min_rate_new, flownumber_new, userpriority_new, path_final, wt_matx, wt_matxreal1, wt_matxreal, blockstate_multi, a11, b11, c11,
                 d11, g11, h11, i11, j11, I, chosenprob1, chosenprob2):
        self.p = p
        self.s_multi = s_multi
        self.d_multi = d_multi
        self.flow_type_multi = flow_type_multi
        self.min_rate_multi = min_rate_multi
        self.flownumber_multi = flownumber_multi
        self.userpriority_multi = userpriority_multi
        self.s_new = s_new
        self.d_new = d_new
        self.flow_type_new = flow_type_new
        self.min_rate_new = min_rate_new
        self.flownumber_new = flownumber_new
        self.userpriority_new = userpriority_new
        self.path_final = path_final
        self.wt_matx = wt_matx
        self.wt_matxreal1 = wt_matxreal1
        self.wt_matxreal = wt_matxreal
        self.blockstate_multi = blockstate_multi
        self.a11 = a11
        self.b11 = b11
        self.c11 = c11
        self.d11 = d11
        self.g11 = g11
        self.h11 = h11
        self.i11 = i11
        self.j11 = j11
        self.I = I
        self.chosenprob1 = chosenprob1
        self.chosenprob2 = chosenprob2

    def service(self):
        check1 = 0
        check2 = 0
        value1 = 0
        value2 = 0
        [m1, n1] = np.shape(a11)
