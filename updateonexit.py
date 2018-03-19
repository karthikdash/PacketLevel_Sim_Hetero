import numpy as np
from call1 import call1
from call2 import call2
from dijkstra import dijkstra

def releaseResources(p, s, d, flow_type, min_rate, flownumber, userpriority,
             flownumber_exit, path_final, wt_matx, wt_matx_real,
             wt_matx_real1, blockstate):
    flownumbersize = np.shape(flownumber)[0]
    index = 0
    for i in range(0, flownumbersize, 1):
        if flownumber[i] == flownumber_exit:
            index = i
    flow_type_exit = flow_type[index]
    min_rate_exit = min_rate[index]
    [path_finalsize1, pathfinalsize] = np.shape(path_final)
    if flow_type_exit == 0:
        for loop in range(0, path_finalsize1, 1):
            if path_final[loop, 0] == flownumber_exit:
                if path_final[loop, 3] == 0:
                    path = path_final[loop, 15:25]
                    cd1 = call2(p, path, flow_type_exit, min_rate_exit, wt_matx,
                                wt_matx_real, wt_matx_real1)
                    cd1.execute()
                    wt_matx = cd1.wt_matx
                    wt_matx_real = cd1.wt_matx_real
                    wt_matx_real1 = cd1.wt_matx_real1

                    path = path_final[loop + 1, 15:25]
                    cd2 = call2(p, path, flow_type_exit, min_rate_exit, wt_matx,
                                wt_matx_real, wt_matx_real1)
                    cd2.execute()
                    wt_matx = cd2.wt_matx
                    wt_matx_real = cd2.wt_matx_real
                    wt_matx_real1 = cd2.wt_matx_real1
                    if loop == path_finalsize1 - 1:
                        path_final[path_finalsize1-1:path_finalsize1, :] = np.zeros((2, p+16))
                    else:
                        path_final[loop:path_finalsize1-2, :] = path_final[loop+2:path_finalsize1, :]
                        path_final[path_finalsize1-1:path_finalsize1, :] = np.zeros((1, p+16))
                    break
                else:
                    path = path_final[loop, 15:25]
                    cd5 = call2(p, path, flow_type_exit, min_rate_exit, wt_matx,
                                wt_matx_real, wt_matx_real1)
                    cd5.execute()
                    wt_matx = cd5.wt_matx
                    wt_matx_real = cd5.wt_matx_real
                    wt_matx_real1 = cd5.wt_matx_real1
                    if loop == path_finalsize1 - 1:
                        path_final[path_finalsize1-1:path_finalsize1, :] = np.zeros((1, p+16))
                    else:
                        path_final[loop:path_finalsize1-1, :] = path_final[loop+1:path_finalsize1, :]
                        path_final[path_finalsize1-1:path_finalsize1, :] = np.zeros((1, p+16))
                    break


    elif flow_type_exit == 1:
        for loop in range(0, path_finalsize1, 1):
            if path_final[loop, 0] == flownumber_exit:
                path = path_final[loop, 11:p+11]
                cd3 = call2(p, path, flow_type_exit, min_rate_exit, wt_matx,
                            wt_matx_real, wt_matx_real1)
                cd3.execute()
                wt_matx = cd3.wt_matx
                wt_matx_real = cd3.wt_matx_real
                wt_matx_real1 = cd3.wt_matx_real1
                if loop == path_finalsize1 - 1:
                    path_final[path_finalsize1-1:path_finalsize1, :] = np.zeros((2, p+15))
                else:
                    path_final[loop:path_finalsize1-2, :] = path_final[loop+2:path_finalsize1, :]
                    path_final[path_finalsize1-1:path_finalsize1, :] = np.zeros((1, p+15))
                break
    elif flow_type_exit == 2:
        for loop in range(0, path_finalsize1, 1):
            if path_final[loop, 0] == flownumber_exit:
                # noofpaths = int(path_final[loop, 3])
                noofpaths = 1
                for loop1 in range(0, noofpaths, 1):
                    min_rate_exit = path_final[loop + loop1, 4]
                    path = path_final[loop + loop1, 15:25]
                    cd4 = call2(p, path, flow_type_exit, min_rate_exit, wt_matx,
                                wt_matx_real, wt_matx_real1)
                    cd4.execute()
                    wt_matx = cd4.wt_matx
                    wt_matx_real = cd4.wt_matx_real
                    wt_matx_real1 = cd4.wt_matx_real1
                path_final[loop:path_finalsize1-noofpaths, :] = path_final[loop+noofpaths:path_finalsize1, :]
                path_final[path_finalsize1-noofpaths:path_finalsize1, :] = np.zeros((noofpaths, p+16))
                break
    if index == flownumbersize and index == 0:
        s = []
        d = []
        min_rate = []
        flow_type = []
        flownumber = []
        userpriority = []
        blockstate = []
    elif index == 0:
        s = s[index+1:flownumbersize]
        d = d[index+1:flownumbersize]
        min_rate = min_rate[index+1:flownumbersize]
        flownumber = flownumber[index+1:flownumbersize]
        flow_type = flow_type[index+1:flownumbersize]
        userpriority = userpriority[index+1:flownumbersize]
        blockstate = blockstate[index+1:flownumbersize]
    elif index == flownumbersize:
        s = s[0:index-1]
        d = d[0:index-1]
        min_rate = min_rate[0:index-1]
        flow_type = flow_type[0:index-1]
        flownumber = flownumber[0:index-1]
        userpriority = userpriority[0:index-1]
        blockstate = blockstate[0:index-1]
    else:
        s = np.delete(s, index)
        d = np.delete(d, index)
        min_rate = np.delete(min_rate, index)
        flow_type = np.delete(flow_type, index)
        flownumber = np.delete(flownumber, index)
        userpriority = np.delete(userpriority,index)
        blockstate = np.delete(blockstate, index)

    return s, d, min_rate, flow_type, flownumber, userpriority, blockstate, path_final, wt_matx, wt_matx_real, wt_matx_real1
