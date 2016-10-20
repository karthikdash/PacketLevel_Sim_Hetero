import numpy as np
import dijkstra from dijkstra
import random


class blocknonreal(object):

    wt_matx = []
    pathsnonreal = []
    __init__(self, s_new, d_new, min_rate_new, wt_matx, eff_capacity_matx):
        self.s_new = s_new
        self.d_new = d_new
        self.min_rate_new = min_rate_new
        self.wt_matx = wt_matx
        self.eff_capacity_matx = eff_capacity_matx

    def execute(object):
        maxsplits = 3
        [p1, p] = np.shape(self.wt_matx)
        dummy = np.divide(1, self.wt_matx)
        dummy1 = np.subtract(dummy, self.min_rate_new)
        dummy2 = dummy1 > 0
        dummy3 = np.multiply(dummy1, dummy2)
        dummy4 = np.divide(1, dummy3)
        for i in range(0, p, 1):
            for j in range(0, p, 1):
                if dummy4[i, j] == -float('inf'):
                    dummy4[i, j] == float('inf')
        # dijkstra3 is not required. dijkstra is enough with more global variables
        dij = dijkstra(self.s_new, self.d_new, dummy4)
        path = dij.execute()
        cost = dij.cost
        visited = dij.visited
        parent = dij.parent
        # End of dijkstra
        paths = np.zeros((p - 1, p + 1))
        count = 1
        for loop in range(0, p, 1):
            path = []
            if parent[loop] != 0:  # If there is a path
                t = loop
                path = loop
                while t != self.s_new:
                    par = parent[t]
                    path = np.append(par, path)
                    t = par
                [pathsize1, pathsize] = np.shape(path)
                if pathsize < p:
                    # Not sure pathsize + 1 or pathsize
                    for j in range(pathsize, p, 1):
                        path[j] = 0
                paths[count, :] = np.append(loop, path)
                count = count + 1
        Anodes2 = []
        for loop in range(0, p, 1):
            if parent[loop] != 0:
                Anodes2 = np.append(Anodes2, loop)
        [Anodes2size1, Anodes2size] = np.shape(Anodes2)
        Anodes1 = Anodes2
        # Might be count
        for loop in range(0, count - 1, 1):
            noofnodes = np.sum(paths[loop, 2:p+1] > 0)
            if noofnodes > 2:
                for loop1 in range(1, noofnodes - 1, 1):
                    node = paths[loop, loop1]
                    index = np.where(Anode2 == node)
                    Anodes1[index] = 0
        Anodes = []
        for loop in range(0, Anodes2size, 1):
            if Anodes1[loop] != 0:
                Anodes = np.append(Anodes, Anodes2[loop])
        # End of forming set a Nodes
        Anodes3 = Anodes
        [Anodes3size1, Anodes3size] = np.shape(Anodes3)
        check = 0
        pathsnonreal = np.zeros((maxsplits, p+1))
        while(Anodes3size > 0):
            countt1 = 1
            completedfrac = 0
            frac = 0.5
            index = random.randit(1, Anodes3size)
            for loop in range(0, count - 1, 1):
                if paths[loop, 1] == Anodes3[inedx]:
                    pathA = paths[loop, 2:p+1]
            flow = np.zeros(p)
            for loop in range(0, p, 1):
                if pathA[loop1+1] != 0:
                    flow[pathA[loop], pathA[loop+1]] = self.min_rate_new
                else:
                    break
            pathAnonreal = []
            for loop in range(0, p - 1, 1):
                if pathA[loop+1] == 0:
                    break
                else:
                    pathAnonreal = np.append(pathAnonreal, pathA[loop])
            dummy5 = np.subtract(dummy, flow)
            while(completedfrac < 1):
                dummy6 = np.subtract(dummy5, frac*self.min_rate_new)
                dummy7 = dummy6 > 0
                dummy8 = np.multiply(dummy6, dummy7)
                dummy9 = np.divide(1, dummy8)
                for loop in range(0, p, 1):
                    for loop1 in range(0, p, 1):
                        if dummy9[loop, loop1] == -float('inf'):
                            dummy9[loop, loop1] == float('inf')
                dij = dijkstra(Anodes3[index], self.d_new, dummy9)
                pathBnonreal = dij.execute()
                [pathBnonrealsize1, pathBnonrealsize] = np.shape(pathBnonreal)
                if pathBnonrealsize == 0 and frac == 0.5:
                    frac = 1.0/3
                    continue
                elif pathBnonrealsize == 0 and frac > 1.0/3:
                    frac = 1.0/3
                    continue
                elif pathBnonrealsize == 0:
                    pathsnonreal = np.zeros((maxsplits, p+1))
                    completedfrac = 0
                    break
                elif frac == 0.5:
                    available = float('inf')
                    for loop in range(0, pathBnonrealsize-1, 1):
                        available = min(available, dummy5[pathBnonreal[loop], pathBnonreal[loop+1]]-0.05*eff_capacity_matx[pathBnonreal[loop], pathBnonreal[loop+1]])
                    frac = available/self.min_rate_new
                    if frac < 1.0/3:
                        break
                    flow1 = np.zeros(p)
                    for loop in range(0, pathBnonrealsize - 1, 1):
                        if pathBnonreal[loop + 1] == 0:
                            break
                        else:
                            flow1[pathBnonreal[loop], pathBnonreal[loop+1]] = frac*self.min_rate_new
                    pathnonreal = np.append(pathAnonreal, pathBnonreal)
                    [pathnonrealsize1, pathnonrealsize] = np.shape(pathnonreal)
                    uniquenodes = np.unique(pathnonreal)  # till now pathnonreal has no zeros in path
                    [uniquenodessize1, uniquenodessize] = np.shape(uniquenodes)
                    for loop1 in range(0, uniquenodessize, 1):
                        repeat = np.where(pathnonreal == uniquienodes[loop1])
                        [repeatsize1, repeatsize] = np.shape(repeat)
                        if repeatsize == 1 or repeatsize == 0:
                            continue
                        else:
                            location1 = repeat[1]
                            location2 = repeat[repeatsize]
                            pathnonreal[location1+1:pathnonrealsize - (location2 - location1)] = pathnonreal[location2+1: pathnonrealsize]
                            pathnonreal = pathnonreal[1:pathnonrealsize-(location2 - location1)]
                            [pathnonrealsize1, pathnonrealsize] = np.shape(pathnonreal)
                    [pathnonrealsize1, pathnonrealsize] = np.shape(pathnonreal)
                    if pathnonrealsize < p:
                        for loop1 in range(pathnonrealsize, p, 1):
                            pathnonreal[loop1] = 0
                    pathsnonreal[count1, :] = np.append(frac*self.min_rate_new, pathnonreal)
                    dummy5 = np.subtract(dummy5, flow1)
                    completedfrac = completedfrac + frac
                    if frac < 0.5:
                        frac = 1.0/3
                    else:
                        frac = 1 - completedfrac
                    count1 = count1 + 1
                else:
                    flow1 = np.zeros(p)
                    for loop in range(0, pathBnonrealsize - 1, 1):
                        if pathBnonreal[loop + 1] == 0:
                            break
                        else:
                            flow1[pathBnonreal[loop], pathBnonreal[loop+1]] = frac*self.min_rate_new
                    pathnonreal = np.append(pathAnonreal, pathBnonreal)
                    [pathnonrealsize1, pathnonrealsize] = np.shape(pathnonreal)
                    uniquenodes = np.unique(pathnonreal)  # till now pathnonreal has no zeros in path
                    [uniquenodessize1, uniquenodessize] = np.shape(uniquenodes)
                    for loop1 in range(0, uniquenodessize, 1):
                        repeat = np.where(pathnonreal == uniquienodes[loop1])
                        [repeatsize1, repeatsize] = np.shape(repeat)
                        if repeatsize == 1 or repeatsize == 0:
                            continue
                        else:
                            location1 = repeat[1]
                            location2 = repeat[repeatsize]
                            pathnonreal[location1+1:pathnonrealsize - (location2 - location1)] = pathnonreal[location2+1: pathnonrealsize]
                            pathnonreal = pathnonreal[1:pathnonrealsize-(location2 - location1)]
                            [pathnonrealsize1, pathnonrealsize] = np.shape(pathnonreal)
                    [pathnonrealsize1, pathnonrealsize] = np.shape(pathnonreal)
                    if pathnonrealsize < p:
                        for loop1 in range(pathnonrealsize, p, 1):
                            pathnonreal[loop1] = 0
                    pathsnonreal[count1, :] = np.append(frac*self.min_rate_new, pathnonreal)
                    dummy5 = np.subtract(dummy5, flow1)
                    completedfrac = completedfrac + frac
                    if completedfrac == 1:
                        check = 1
                        for loop1 in range(0, maxsplits, 1):
                            if pathsnonreal[loop1, 1] == 0:
                                break
                            else:
                                pathresourceupdate = pathsnonreal[loop1, 2:p+1]
                                [pathresourceupdatesize1, pathresourceupdatesize] = np.shape(pathresourceupdate)
                                for loop2 in range(0, pathresourceupdatesize - 1, 1):
                                    if pathresourceupdate[loop2] != 0 and pathresourceupdate[loop2+1] != 0:
                                        dummy[pathresourceupdate[loop2], pathresourceupdate[loop2+1]] = dummy[pathresourceupdate[loop2], pathresourceupdate[loop2+1]] - pathsnonreal[loop1, 1]
                                    else:
                                        break

                        self.wt_matx = np.divide(1, dummy)
                        break
                    if completedfrac == 1.0/3 or completedfrac == 2.0/3:
                        frac = 1.0/3
                    else:
                        frac = 1 - completedfrac
                    count1 = count1 + 1
            if check == 1:
                break
            Anodes3[index] = []
            [Anodes3size1, Anodes3size] = np.shape(Anodes3)
        self.pathsnonreal = pathsnonreal
