import numpy as np
from call1 import call1
from call2 import call2
from dijkstra import dijkstra


class updateonexit(object):
    def __init__(self, p, s, d, flow_type, min_rate, flownumber, userpriority,
                 flownumber_exit, path_final, wt_matx, wt_matx_real,
                 wt_matx_real1, blockstate):
        self.p = p
        self.s = s
        self.d = d
        self.flow_type = flow_type
        self.min_rate = min_rate
        self.flownumber = flownumber
        self.userpriority = userpriority
        self.flownumber_exit = flownumber_exit
        self.path_final = path_final
        self.wt_matx = wt_matx
        self.wt_matx_real = wt_matx_real
        self.wt_matx_real1 = wt_matx_real1
        self.blockstate = blockstate

    def execute(self):
        p = self.p
        flownumbersize = np.shape(self.flownumber)[0]
        index = 0
        for i in range(0, flownumbersize, 1):
            if self.flownumber[i] == self.flownumber_exit:
                index = i
        flow_type_exit = self.flow_type[index]
        min_rate_exit = self.min_rate[index]
        [path_finalsize1, pathfinalsize] = np.shape(self.path_final)
        if flow_type_exit == 0:
            for loop in range(0, path_finalsize1, 1):
                if self.path_final[loop, 0] == self.flownumber_exit:
                    path = self.path_final[loop, 11:p+11]
                    cd1 = call2(self.p, path, flow_type_exit, min_rate_exit, self.wt_matx,
                                self.wt_matx_real, self.wt_matx_real1)
                    cd1.execute()
                    self.wt_matx = cd1.wt_matx
                    self.wt_matx_real = cd1.wt_matx_real
                    self.wt_matx_real1 = cd1.wt_matx_real1

                    path = self.path_final[loop + 1, 11:p+11]
                    cd2 = call2(self.p, path, flow_type_exit, min_rate_exit, self.wt_matx,
                                self.wt_matx_real, self.wt_matx_real1)
                    cd2.execute()
                    self.wt_matx = cd2.wt_matx
                    self.wt_matx_real = cd2.wt_matx_real
                    self.wt_matx_real1 = cd2.wt_matx_real1
                    if loop == path_finalsize1 - 1:
                        self.path_final[path_finalsize1-1:path_finalsize1, :] = np.zeros((2, p+11))
                    else:
                        self.path_final[loop:path_finalsize1-2, :] = self.path_final[loop+2:path_finalsize1, :]
                        self.path_final[path_finalsize1-1:path_finalsize1, :] = np.zeros((1, p+11))
                    break
        elif flow_type_exit == 1:
            for loop in range(0, path_finalsize1, 1):
                if self.path_final[loop, 0] == self.flownumber_exit:
                    path = self.path_final[loop, 11:p+11]
                    cd3 = call2(self.p, path, flow_type_exit, min_rate_exit, self.wt_matx,
                                self.wt_matx_real, self.wt_matx_real1)
                    cd3.execute()
                    self.wt_matx = cd3.wt_matx
                    self.wt_matx_real = cd3.wt_matx_real
                    self.wt_matx_real1 = cd3.wt_matx_real1
                    if loop == path_finalsize1 - 1:
                        self.path_final[path_finalsize1-1:path_finalsize1, :] = np.zeros((2, p+11))
                    else:
                        self.path_final[loop:path_finalsize1-2, :] = self.path_final[loop+2:path_finalsize1, :]
                        self.path_final[path_finalsize1-1:path_finalsize1, :] = np.zeros((1, p+11))
                    break
        elif flow_type_exit == 2:
            for loop in range(0, path_finalsize1, 1):
                if self.path_final[loop, 0] == self.flownumber_exit:
                    # noofpaths = int(self.path_final[loop, 3])
                    noofpaths = 1
                    for loop1 in range(0, noofpaths, 1):
                        min_rate_exit = self.path_final[loop + loop1, 4]
                        path = self.path_final[loop + loop1 , 11:p+11]
                        cd4 = call2(self.p, path, flow_type_exit, min_rate_exit, self.wt_matx,
                                    self.wt_matx_real, self.wt_matx_real1)
                        cd4.execute()
                        self.wt_matx = cd4.wt_matx
                        self.wt_matx_real = cd4.wt_matx_real
                        self.wt_matx_real1 = cd4.wt_matx_real1
                    self.path_final[loop:path_finalsize1-noofpaths, :] = self.path_final[loop+noofpaths:path_finalsize1, :]
                    self.path_final[path_finalsize1-noofpaths:path_finalsize1, :] = np.zeros((noofpaths, p+11))
                    break
        if index == flownumbersize and index == 0:
            self.s = []
            self.d = []
            self.min_rate = []
            self.flow_type = []
            self.flownumber = []
            self.userpriority = []
            self.blockstate = []
        elif index == 0:
            self.s = self.s[index+1:flownumbersize]
            self.d = self.d[index+1:flownumbersize]
            self.min_rate = self.min_rate[index+1:flownumbersize]
            self.flownumber = self.flownumber[index+1:flownumbersize]
            self.flow_type = self.flow_type[index+1:flownumbersize]
            self.userpriority = self.userpriority[index+1:flownumbersize]
            self.blockstate = self.blockstate[index+1:flownumbersize]
        elif index == flownumbersize:
            self.s = self.s[0:index-1]
            self.d = self.d[0:index-1]
            self.min_rate = self.min_rate[0:index-1]
            self.flow_type = self.flow_type[0:index-1]
            self.flownumber = self.flownumber[0:index-1]
            self.userpriority = self.userpriority[0:index-1]
            self.blockstate = self.blockstate[0:index-1]
        else:
            '''
            self.s = np.append(self.s[0:index], self.s[index:flownumbersize-1])
            self.d = np.append(self.d[0:index], self.d[index:flownumbersize-1])
            self.min_rate = np.append(self.min_rate[0:index], self.min_rate[index:flownumbersize-1])
            self.flow_type = np.append(self.flow_type[0:index], self.flow_type[index:flownumbersize-1])
            self.flownumber = np.append(self.flownumber[0:index], self.flownumber[index:flownumbersize-1])
            self.userpriority = np.append(self.userpriority[0:index], self.userpriority[index:flownumbersize-1])
            self.blockstate = np.append(self.blockstate[0:index], self.blockstate[index:flownumbersize-1])
            '''
            self.s = np.delete(self.s, index)
            self.d = np.delete(self.d, index)
            self.min_rate = np.delete(self.min_rate, index)
            self.flow_type = np.delete(self.flow_type, index)
            self.flownumber = np.delete(self.flownumber, index)
            self.userpriority = np.delete(self.userpriority,index)
            self.blockstate = np.delete(self.blockstate, index)