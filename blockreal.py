import numpy as np
import dijkstra from dijkstra
import random


class blockreal(object):
    __init__(self, s_new, d_new, min_rate_new, path_final, wt_matx_real, wt_matx_real1, wt_matx):
        self.s_new = s_new
        self.d_new = d_new
        self.min_rate_new = min_rate_new
        self.path_final = path_final
        self.wt_matx_real = wt_matx_real
        self.wt_matx_real1 = wt_matx_real1
        self.wt_matx = wt_matx

    def execute(object):
        [p1, p] = np.shape(self.wt_matx)
        with np.errstate(divide='ignore', invalid='ignore'):
            dummy1 = np.divide(1, self.wt_matx_real1)
            dummy2 = np.subtract(1, self.min_rate_new)
            dummy3 = dummy2 > 0
            dummy4 = np.divide(1, self.wt_matx)
            dummy5 = np.multiply(dummy4, dummy3)
            dummy6 = np.subtract(dummy5, self.min_rate_new)
            dummy7 = dummy6 > 0
            dummy8 = np.multiply(dummy6, dummy7)
            dummy9 = np.divide(1, dummy8)
        for o in range(0, p, 1):
            for j in range(0, p, 1):
                if dummy9[o][j] == -float('inf'):
                    dummy9[o][j] == float('inf')
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
        [mPF, nPF] = np.shape(self.path_final)
        check = 0
        while Anodes3size > 0:
            index = random.randit(1, Anodes3size)
            # For Destination Node First
            if dummy3[Anodes3[index], self.d_new] > 0:
                # Routing Real-time flow
                for loop1 in range(0, p-1, 1):
                    if paths[loop1, 1] == Anodes3[index]:
                        pathA = paths[loop1][2:np.where(paths[loop1, 2:p+1] == Anodes3[index])+1]
                self.pathreal = np.append(pathA, self.d_new)
                [self.pathrealsize1, self.pathrealsize] = np.shape(self.pathreal)
                if self.pathrealsize < p:
                    for loop1 in range(self.pathrealsize, p, 1):
                        self.pathreal[loop1] = 0
                # End of Routing Real Time flow
                # Updating weight matrix after Routing Real-time flow
                flow = np.zeros((p))  # No of Nodes
                for loop1 in range(0, p-1, 1):
                    if self.pathreal[loop1+1] == 0:
                        break
                    else:
                        flow[self.pathreal[loop1]][self.pathreal[loop1+1]] = self.min_rate_new
                dummy12 = np.subtract(dummy4, flow)
                dummy13 = np.divide(1, dummy12)  # All Positive
                dummy13[Anodes3[index]][self.d_new] = float('inf')  # Weight Matrix after including new real time flow
                # End of updating weight matrix after routing real-time flow
                # Collecting path numbers of non-real time flows
                PF1 = []
                rate = []
                decr = 0
                for PF in range(0, mPF, 1):
                    if decr > 0:
                        decr = decr - 1
                        continue
                    # Might be PF, 1
                    if self.path_final[PF, 0] == 0:
                        break
                    else if self.path_final[PF, 2] == 2 and self.path_final[PF, 4] == 1:
                        for k in (5, nPf - 1, 1):
                            if self.path_final[Pf, k] == Anodes3[index] and path_final[PF, k+1] == d_new:
                                # Vertical Concatenation
                                PF1 = np.concatenate((PF, PF, 0, 0), axis=1)
                                rate = np.append(rate, self.path_final[PF, 5])
                                break
                            else if self.path_final[PF, k+1] == 0:
                                break
                    else if self.path_final[PF, 2] == 2 and self.path_final[PF, 4] > 1:
                        decr = self.path_final[PF, 4] - 1
                        rateonlink = 0
                        PF2 = np.zeros((1, 3))
                        for loop1 in range(0, self.path_final[PF, 4]):
                            for k in range(5, nPF - 1, 1):
                                if self.path_final[PF + loop1 - 1, k] == Anodes3[index] and self.path_final[PF + loop1 - 1, k + 1] == self.d_new:
                                    rateonlink = rateonlink + self.path_final[PF+loop1-1, 5]
                                    PF2[loop1] = PF + loop1 - 1
                                    break
                                else if self.path_final[PF+loop1-1, k+1] == 0:
                                    PF2[loop1] = 0
                                    break
                        if rateonlink > 0:
                            # Vertical Concatenation
                            PF1 = np.concatenate((PF1, PF2), axis=1)
                            rate = np.append(rate, rateonlink)
                # End of Collecting Path numbers of Non-Real Time flows
                # Finding Paths of Non-Real time flow
                [row1, column1] = np.shape(PF1)
                rate2 = np.multiply(rate > self.min_rate_new, rate)
                for loop1 in range(0, row1, 1):
                    if rate2[loop1] == 0:
                        rate[loop1] = float('inf')
                if np.sum(rate > self.min_rate_new) > 0:  # To check if there are any non-real tim flows having requirement grater than non-realtime flows
                    # Check if c and I are proper
                    [c, I] = np.nanmin(rate2)
                    if c < float('inf'):
                        dummy15 = np.subtract(np.divide(1, dummy13), c)
                        dummy16 = dummy15 > 0
                        dummy17 = np.multiply(dummy15, dummy16)
                        dummy18 = np.divide(1, dummy17)
                        for o in range(0, p, 1):
                            for j in range(0, p, 1):
                                if dummy18[o, j] == -float('inf'):
                                    dummy18[o, j] = float('inf')
                        dij = dijkstra(Anodes3[index], self.d_new, dummy18)
                        path5 = dij.execute()
                        [row2, column2] = np.shape(path5)
                        check1 = 0
                        for loop1 in range(0, 3, 1):
                            if PF1[I, loop1] > 0:
                                check1 = check1 or self.path_final[PF1[I][loop1]][4] > 1
                        if column2 == 0 and check1 != 1:
                            dummy15 = np.subtract(np.divide(1, dummy13), self.min_rate_new)
                            dummy16 = dummy15 > 0
                            dummy17 = np.multiply(dummy15, dummy16)
                            dummy18 = np.divide(1, dummy17)
                            for o in range(0, p, 1):
                                for j in range(0, p, 1):
                                    if dummy18[o, j] == -float('inf'):
                                        dummy18[o, j] = float('inf')
                            dij = dijkstra(Anodes3[index], self.d_new, dummy18)
                            path5 = dij.execute()
                            [row3, column3] = np.shape(path5)
                            if column3 != 0:
                                PF3 = PF1[I, :]
                                path6 = []
                                for loop in range(5, nPF, 1):
                                    # Might be PF[1]
                                    if self.path_final[PF3[0], loop1] == Anodes3[index]:
                                        path6 = np.append(path6, path5)
                                        [path6size1, path6size] = np.shape(path6)
                                        path6 = path6[1:path6size-1]
                                    else if self.path_final[PF3[0], loop1] == 0:
                                        break
                                    else:
                                        path6 = np.append(path6, self.path_final[PF3[0], loop1])
                                [path6size1, path6size] = np.shape(path6)
                                uniquienodes = np.unique(path6)
                                [uniquienodessize1, uniquienodessize] = np.shape(uniquienodes)
                                for loop2 in range(0, uniquienodessize, 1):
                                    repeat = np.where(path6 == uniquienodes[loop2])
                                    [repeatsize1, repeatsize] = np.shape(repeat)
                                    if repeatsize == 1 or repeatsize == 0:
                                        continue
                                    else:
                                        location1 = repeat[1]
                                        location2 = repeat[repeatsize]
                                        path6[location1+1:path6size - (location2 - location1)] = path6[location2+1: path6size]
                                        path6 = path6[1:path6size-(location2 - location1)]
                                        [path6size1, path6size] = np.shape(path6)
                                [path6size1, path6size] = np.shape(path6)
                                if path6size < p:
                                    for loop2 in range(path6size, p, 1):
                                        path6[loop2] = 0
                                path7 = path6
                                # Not Sure about the indices
                                pathrelease = self.path_final[PF3[0], 6:nPF]
                                [pathreleasesize1, pathreleasesize] = np.shape(pathrelease)
                                for loop1 in range(0, pathreleasesize - 1, 1):
                                    if pathrelease[loop1] != 0 and pathrelease[loop1 + 1] != 0:
                                        dummy12[pathrelease[loop1], pathrelease[loop1 + 1]] = dummy12[pathrelease[loop1, pathrelease[loop1 + 1] + self.min_rate_new]]
                                    else:
                                        break
                                for loop3 in range(0, path6size - 1, 1):
                                    if path6[loop3] != 0 and path6[loop3 + 1] != 0:
                                        dummy12[path6[loop3], path6[loop3 + 1]] = dummy12[path6[loop3, path6[loop3 + 1] - self.min_rate_new]]
                                    else:
                                        break
                                if self.path_final[mPF, :] = np.zeros((1, p+5)):
                                    self.path_final[PF3[1]+2:mPF, :] = self.path_final[PF3[1]+1:mPF-1, :]
                                else:
                                    self.path_final[mPF+1, :] = self.path_final[mPF, :]
                                    self.path_final[PF3[1]+2:mPF, :] = self.path_final[PF3[1]+1:mPF-1, :]
                                self.path_final[PF3[1], 4] = 2
                                self.path_final[PF3[1], 5] = self.path_final[PF3[1], 5] - self.min_rate_new
                                self.path_final[PF3[1]+1, 1:4] = self.path_final[PF3[1], 1:4]
                                self.path_final[PF3[1]+1, 5] = self.min_rate_new
                                self.path_final[PF3[1]+1, 6:nPF] = path7
                                self.wt_matx = np.divide(1, dummy12)
                                self.wt_matx_real = np.divide(1, self.wt_matx_real - flow)
                                self.wt_matx_real1 = np.divide(1, self.wt_matx_real1 - flow)
                                break
                            # End of finding paths of Non-Real time flow
            # End of destination node first
            else:
                for loop in range(0, p, 1):
                    # Routing Real-time Flow
                    if dummy3[Anodes3[index]m loop] > 0 and loop != self.d_new:
                        dij = dijkstra(self.s_new, self.d_new, dummy4)
                        path = dij.execute()
                        [pathsize1, pathsize] = np.shape(path)
                        if pathsize == 0:
                            continue
                        else:
                            for loop1 in range(0, p-1, 1):
                                if paths[loop1, 1] == Anodes3[index]:
                                    pathA = paths[loop1][2:np.where(paths[loop1, 2:p+1] == Anodes[index])+1]
                                    break
                            self.pathreal = np.append(pathA, path)
                            [self.pathrealsize1, self.pathrealsize] = np.shape(self.pathreal)
                            if self.pathrealsize < p:
                                for loop1 in range(self.pathrealsize, p, 1):
                                    self.pathreal[loop1] = 0
                            # End of routing real-time flowarrivaltime
                            # Updating weight matrix after routing real-time flow
                            flow = np.zerps(p)
                            for loop2 in range(0, p-1, 1):
                                if self.pathreal[loop1 + 1] == 0:
                                    break
                                else:
                                    flow[self.pathreal[loop1], self.pathreal[loop1 + 1]] = self.min_rate_new
                            dummy12 = np.subtract(dummy4, flow)
                            dummy13 = np.divide(1, dummy12)
                            dummy13[Anodes3[index], loop] = float('inf')  # Weight matrix after including realtime flow
                            # End of updating weight matrix after routing realtime flow
                            # Collecting path numbers of non-real time flows
                            PF1 = []
                            rate = []
                            decr = 0
                            for PF in range(0, mPF, 1):
                                if decr > 0:
                                    decr = decr - 1
                                    continue
                                if self.path_final[PF, 1] == 0:
                                    break
                                else if self.path_fina[PF, 2] == 2 and self.path_final[Pf, 4] == 1:
                                    for k in range(5, nPF - 1, 1):
                                        if self.path_final[PF, k] == Anodes3[index] and self.path_final(PF, k+1) == loop:
                                            PF1 = np.concatenate((PF1, PF, 0, 0), axis=1)
                                            rate = np.append(rate, self.path_final[PF, 5])
                                            break
                                else if self.path_final[PF, 2] == 2 and self.path_final[PF, 4] > 1:
                                        decr = self.path_final[PF, 4] - 1
                                        rateonlink = 0
                                        PF2 = np.zeros((1, 3))
                                        for loop1 in range(0, self.path_final[PF, 4], 1):
                                            for k in range(5, nPF-1, 1):
                                                if self.path_final[PF+loop1, k] == Anodes3[index] and self.path_final(PF+loop1-1, k+1) == loop:
                                                    rateonlink = rateonlink + self.path_final[PF+loop1-1, 5]
                                                    PF2[loop1] = PF + loop1 - 1
                                                    break
                                                else if self.path_final[PF+loop1-1, k+1] == 0:
                                                    PF2[loop1] = 0
                                                    break
                                        if rateonlink > 0:
                                            PF1 = np.concatenate((PF1, PF2), axis=1)
                                            rate = np.append(rate, rateonlink)
                            # End of collecting Path numbers of non-real time flows
                            # Finding paths of non-real time flow
                            [row1, column1] = np.shape(PF1)
                            rate2 = np.multiply(rate > self.min_rate_new, rate)
                            for loop1 in range(0, row1, 1):
                                if rate2[loop1] == 0:
                                    rate[loop1] = float('inf')
                            if np.sum(rate > self.min_rate_new) > 0:  # To check if there are any non-real tim flows having requirement grater than non-realtime flows
                                # Check if c and I are proper
                                [c, I] = np.nanmin(rate2)
                                dummy15 = np.subtract(np.divide(1, dummy13), c)
                                dummy16 = dummy15 > 0
                                dummy17 = np.multiply(dummy15, dummy16)
                                dummy18 = np.divide(1, dummy17)
                                for o in range(0, p, 1):
                                    for j in range(0, p, 1):
                                        if dummy18[o, j] == -float('inf'):
                                            dummy18[o, j] = float('inf')
                                dij = dijkstra(Anodes3[index], self.d_new, dummy18)
                                path5 = dij.execute()
                                [row2, column2] = np.shape(path5)
                                check1 = 0
                                for loop1 in range(0, 3, 1):
                                    if PF1[I, loop1] > 0:
                                        check1 = check1 or self.path_final[PF1[I][loop1]][4] > 1
                                if column2 == 0 and check1 == 1:
                                    continue
                                else if column2 == 0:
                                    dummy15 = np.subtract(np.divide(1, dummy13), self.min_rate_new)
                                    dummy16 = dummy15 > 0
                                    dummy17 = np.multiply(dummy15, dummy16)
                                    dummy18 = np.divide(1, dummy17)
                                    for o in range(0, p, 1):
                                        for j in range(0, p, 1):
                                            if dummy18[o, j] == -float('inf'):
                                                dummy18[o, j] = float('inf')
                                    dij = dijkstra(Anodes3[index], self.d_new, dummy18)
                                    path5 = dij.execute()
                                    [row3, column3] = np.shape(path5)
                                    if column3 == 0:
                                        continue
                                    else:
                                        PF3 = PF1[I, :]
                                        path6 = []
                                        for loop in range(5, nPF, 1):
                                            # Might be PF[1]
                                            if self.path_final[PF3[0], loop1] == Anodes3[index]:
                                                path6 = np.append(path6, path5)
                                                [path6size1, path6size] = np.shape(path6)
                                                path6 = path6[1:path6size-1]
                                            else if self.path_final[PF3[0], loop1] == 0:
                                                break
                                            else:
                                                path6 = np.append(path6, self.path_final[PF3[0], loop1])
                                        [path6size1, path6size] = np.shape(path6)
                                        uniquienodes = np.unique(path6)
                                        [uniquienodessize1, uniquienodessize] = np.shape(uniquienodes)
                                        for loop2 in range(0, uniquienodessize, 1):
                                            repeat = np.where(path6 == uniquienodes[loop2])
                                            [repeatsize1, repeatsize] = np.shape(repeat)
                                            if repeatsize == 1 or repeatsize == 0:
                                                continue
                                            else:
                                                location1 = repeat[1]
                                                location2 = repeat[repeatsize]
                                                path6[location1+1:path6size - (location2 - location1)] = path6[location2+1: path6size]
                                                path6 = path6[1:path6size-(location2 - location1)]
                                                [path6size1, path6size] = np.shape(path6)
                                        [path6size1, path6size] = np.shape(path6)
                                        if path6size < p:
                                            for loop2 in range(path6size, p, 1):
                                                path6[loop2] = 0
                                        path7 = path6
                                        # Not Sure about the indices
                                        pathrelease = self.path_final[PF3[0], 6:nPF]
                                        [pathreleasesize1, pathreleasesize] = np.shape(pathrelease)
                                        for loop1 in range(0, pathreleasesize - 1, 1):
                                            if pathrelease[loop1] != 0 and pathrelease[loop1 + 1] != 0:
                                                    dummy12[pathrelease[loop1], pathrelease[loop1 + 1]] = dummy12[pathrelease[loop1, pathrelease[loop1 + 1] + self.min_rate_new]]
                                            else:
                                                break
                                        for loop3 in range(0, path6size - 1, 1):
                                            if path6[loop3] != 0 and path6[loop3 + 1] != 0:
                                                dummy12[path6[loop3], path6[loop3 + 1]] = dummy12[path6[loop3, path6[loop3 + 1] - self.min_rate_new]]
                                            else:
                                                break
                                        if self.path_final[mPF, :] = np.zeros((1, p+5)):
                                            self.path_final[PF3[1]+2:mPF, :] = self.path_final[PF3[1]+1:mPF-1, :]
                                        else:
                                            self.path_final[mPF+1, :] = self.path_final[mPF, :]
                                            self.path_final[PF3[1]+2:mPF, :] = self.path_final[PF3[1]+1:mPF-1, :]
                                        self.path_final[PF3[1], 4] = 2
                                        self.path_final[PF3[1], 5] = self.path_final[PF3[1], 5] - self.min_rate_new
                                        self.path_final[PF3[1]+1, 1:4] = self.path_final[PF3[1], 1:4]
                                        self.path_final[PF3[1]+1, 5] = self.min_rate_new
                                        self.path_final[PF3[1]+1, 6:nPF] = path7
                                else:
                                    PF3 = PF1[I, :]
                                    for loop1 in range(0, 3, 1):
                                        if PF3[loop1] > 0:
                                            path6 = []
                                            for loop2 in range(5, nPF, 1):
                                                # Might be PF[1]
                                                if self.path_final[PF3[[loop1, loop2]]] == Anodes3[index]:
                                                    path6 = np.append(path6, path5)
                                                    [path6size1, path6size] = np.shape(path6)
                                                    path6 = path6[1:path6size-1]
                                                else if self.path_final[PF3[[loop1, loop2]]] == 0:
                                                    break
                                                else:
                                                    path6 = np.append(path6, self.path_final[PF3[[loop1, loop2]]])
                                            [path6size1, path6size] = np.shape(path6)
                                            uniquienodes = np.unique(path6)
                                            [uniquienodessize1, uniquienodessize] = np.shape(uniquienodes)
                                            for loop2 in range(0, uniquienodessize, 1):
                                                repeat = np.where(path6 == uniquienodes[loop2])
                                                [repeatsize1, repeatsize] = np.shape(repeat)
                                                if repeatsize == 1 or repeatsize == 0:
                                                    continue
                                                else:
                                                    location1 = repeat[1]
                                                    location2 = repeat[repeatsize]
                                                    path6[location1+1:path6size - (location2 - location1)] = path6[location2+1: path6size]
                                                    path6 = path6[1:path6size-(location2 - location1)]
                                                    [path6size1, path6size] = np.shape(path6)
                                            [path6size1, path6size] = np.shape(path6)
                                            if path6size < p:
                                                for loop2 in range(path6size, p, 1):
                                                    path6[loop2] = 0
                                            path7 = path6
                                            # Not Sure about the indices
                                            pathrelease = self.path_final[PF3[loop1], 6:nPF]
                                            [pathreleasesize1, pathreleasesize] = np.shape(pathrelease)
                                            for loop3 in range(0, pathreleasesize - 1, 1):
                                                if pathrelease[loop3] != 0 and pathrelease[loop3 + 1] != 0:
                                                        dummy12[pathrelease[loop3], pathrelease[loop3 + 1]] = dummy12[pathrelease[loop3, pathrelease[loop3 + 1] + self.path_final[PF3[loop1], 5]]]
                                                else:
                                                    break
                                            for loop3 in range(0, path6size, 1):
                                                if path6[loop3] != 0 and path6[loop3 + 1] != 0:
                                                    dummy12[path6[loop3], path6[loop3 + 1]] = dummy12[path6[loop3, path6[loop3 + 1] - self.path_final[PF3[loop1], 5]]]
                                                else:
                                                    break
                                            self.path_final[PF3[loop1], 6:nPF] = path7
                                self.wt_matx = np.divide(1, dummy12)
                                self.wt_matx_real = np.divide(1, self.wt_matx_real - flow)
                                self.wt_matx_real1 = np.divide(1, self.wt_matx_real1 - flow)
                                check = 1
                                break
                            else:
                                continue
            if check = 1:
                break
            else:
                Anodes[index] = []
                [Anodes3size1, Anodes3size] = np.shape(Anodes3)
        if Anodes3size == 0:
            self.pathreal = np.zeros((1, p))
