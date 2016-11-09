import numpy as np
import random

class allocaterealupdated1(object):

    def __init__(self, p, s_multi, d_multi, flow_type_multi, min_rate_multi, flownumber_multi, userpriority_multi, s_new, d_new, flow_type_new,
                 min_rate_new, flownumber_new, userpriority_new, path_final, wt_matx, wt_matx_real1, wt_matx_real, blockstate_multi, a11, b11, c11,
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
        self.wt_matx_real1 = wt_matx_real1
        self.wt_matx_real = wt_matx_real
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
        [m1, n1] = np.shape(self.a11)
        [m2, n2] = np.shape(self.b11)
        [m3, n3] = np.shape(self.c11)
        [m4, n4] = np.shape(self.d11)
        pathsno = [m1, m2, m3, m4]
        if self.I <= 3:
            chosenprob1index = np.arange(0, pathsno[self.I])
            chosenprob2index = np.arange(0, pathsno[self.I])
            while value1 == 0:
                [chosenprob1size1, chosenprob1size] = np.shape(self.chosenprob1)
                if chosenprob1size == 0:
                    check1 = 1
                    break
                self.chosenprob1 = self.chosenprob1 / sum(self.chosenprob1)
                chosenprob12 = np.zeros((chosenprob1size1, 1))
                for loop in range(0, chosenprob1size1, 1):
                    if loop == 0:
                        chosenprob12[0] = self.chosenprob1[0]
                    else:
                        chosenprob12[loop] = chosenprob12[loop - 1] + self.chosenprob1[loop]
                onezeros1 = random.random()
                for z in range(0, len(chosenprob12) - 1, 1):
                    if chosenprob12[z] > onezeros1:
                        index1 = z
                        value1 = np.ceil(chosenprob12[z])[0]
                        break
                if self.I == 0:  # Voice Type
                    path1 = self.a11[chosenprob1index[index1], :]
                elif self.I == 1:
                    path1 = self.b11[chosenprob1index[index1], :]
                elif self.I == 2:
                    path1 = self.c11[chosenprob1index[index1], :]
                elif self.I == 3:
                    path1 = self.d11[chosenprob1index[index1], :]
                path1size = np.shape(path1)[0]
                for i in range(0, path1size - 1, 1):
                    if path1[i + 1] != 0:
                        if (1 / self.wt_matx_real1[path1[i] - 1, path1[i + 1] - 1]) - self.min_rate_new < 0:
                            value1 = 0
                            self.chosenprob1 = np.delete(self.chosenprob1, index1, axis=0)
                            chosenprob1index = np.delete(chosenprob1index, index1, axis=0)
                            break
                        elif (1 / self.wt_matx[path1[i] - 1, path1[i + 1] - 1]) - self.min_rate_new < 0:
                            value1 = 0
                            self.chosenprob1 = np.delete(self.chosenprob1, index1, axis=0)
                            chosenprob1index = np.delete(chosenprob1index, index1, axis=0)
                            break
                    else:
                        break
                if value1 == 1:
                    break

            if check1 != 1:
                while value2 == 0:
                    [chosenprob2size1, chosenprob2size] = np.shape(self.chosenprob2)
                    if chosenprob2size == 0:
                        check2 = 1
                        break
                    self.chosenprob2 = self.chosenprob2 / sum(self.chosenprob2)
                    chosenprob22 = np.zeros((chosenprob2size1, 1))
                    for loop in range(0, chosenprob2size1, 1):
                        if loop == 0:
                            chosenprob22[0] = self.chosenprob2[0]
                        else:
                            chosenprob22[loop] = chosenprob22[loop - 1] + self.chosenprob2[loop]
                    onezeros2 = random.random()
                    for z in range(0, len(chosenprob22) - 1, 1):
                        if chosenprob22[z] > onezeros2:
                            index2 = z
                            value2 = np.ceil(chosenprob22[z])[0]
                            break
                    if self.I == 0:  # Voice Type
                        path2 = self.g11[chosenprob2index[index2], :]
                    elif self.I == 1:
                        path2 = self.h11[chosenprob2index[index2], :]
                    elif self.I == 2:
                        path2 = self.i11[chosenprob2index[index2], :]
                    elif self.I == 3:
                        path2 = self.j11[chosenprob2index[index2], :]
                    path2size = np.shape(path2)[0]
                    for i in range(0, path2size - 1, 1):
                        if path2[i + 1] != 0:
                            if (1 / self.wt_matx_real1[path2[i] - 1, path2[i + 1] - 1]) - self.min_rate_new < 0:
                                value2 = 0
                                self.chosenprob2 = np.delete(self.chosenprob2, index2, axis=0)
                                chosenprob2index = np.delete(chosenprob2index, index2, axis=0)
                                break
                            elif (1 / self.wt_matx[path2[i] - 1, path2[i + 1] - 1]) - self.min_rate_new < 0:
                                value2 = 0
                                self.chosenprob2 = np.delete(self.chosenprob2, index2, axis=0)
                                chosenprob2index = np.delete(chosenprob2index, index2, axis=0)
                                break
                        else:
                            break
                    if value2 == 1:
                        break
            else:
                check2 = 1
        elif self.I <= 11:
            chosenprob1index = np.arange(0, pathsno[self.I-4])
            chosenprob2index = np.arange(0, pathsno[self.I-4])
            while value1 == 0:
                [chosenprob1size1, chosenprob1size] = np.shape(self.chosenprob1)
                if chosenprob1size1 == 0:
                    check1 = 1
                    break
                self.chosenprob1 = self.chosenprob1 / sum(self.chosenprob1)
                chosenprob12 = np.zeros((chosenprob1size1, 1))
                for loop in range(0, chosenprob1size1, 1):
                    if loop == 0:
                        chosenprob12[0] = self.chosenprob1[0]
                    else:
                        chosenprob12[loop] = chosenprob12[loop - 1] + self.chosenprob1[loop]
                onezeros1 = random.random()
                for z in range(0, len(chosenprob12) - 1, 1):
                    if chosenprob12[z] > onezeros1:
                        index1 = z
                        value1 = np.ceil(chosenprob12[z])[0]
                        break
                if self.I == 4:  # Voice Type
                    if len(chosenprob1index) <= index1:
                        index1 = index1 - 1
                        value1 = 1
                    path1 = self.a11[chosenprob1index[index1], :]
                elif self.I == 5:
                    if len(chosenprob1index) <= index1:
                        index1 = index1 - 1
                        value1 = 1
                    path1 = self.b11[chosenprob1index[index1], :]
                elif self.I == 6:
                    if len(chosenprob1index) <= index1:
                        index1 = index1 - 1
                        value1 = 1
                    path1 = self.c11[chosenprob1index[index1], :]
                elif self.I == 7:
                    if len(chosenprob1index) <= index1:
                        index1 = index1 - 1
                        value1 = 1
                    path1 = self.d11[chosenprob1index[index1], :]
                path1size = np.shape(path1)[0]
                for i in range(0, path1size - 1, 1):
                    if path1[i + 1] != 0:
                        if (1 / self.wt_matx_real1[path1[i] - 1, path1[i + 1] - 1]) - self.min_rate_new < 0:
                            value1 = 0
                            self.chosenprob1 = np.delete(self.chosenprob1, index1, axis=0)
                            chosenprob1index = np.delete(chosenprob1index, index1, axis=0)
                            break
                        elif (1 / self.wt_matx[path1[i] - 1, path1[i + 1] - 1]) - self.min_rate_new < 0:
                            value1 = 0
                            self.chosenprob1 = np.delete(self.chosenprob1, index1, axis=0)
                            chosenprob1index = np.delete(chosenprob1index, index1, axis=0)
                            break
                    else:
                        break
                if value1 == 1:
                    break

            if check1 != 1:
                while value2 == 0:
                    [chosenprob2size1, chosenprob2size] = np.shape(self.chosenprob2)
                    if chosenprob2size == 0:
                        check2 = 1
                        break
                    self.chosenprob2 = self.chosenprob2 / sum(self.chosenprob2)
                    chosenprob22 = np.zeros((chosenprob2size1, 1))
                    for loop in range(0, chosenprob2size1, 1):
                        if loop == 0:
                            chosenprob22[0] = self.chosenprob2[0]
                        else:
                            chosenprob22[loop] = chosenprob22[loop - 1] + self.chosenprob2[loop]
                    onezeros2 = random.random()
                    for z in range(0, len(chosenprob22) - 1, 1):
                        if chosenprob22[z] > onezeros2:
                            index2 = z
                            value2 = np.ceil(chosenprob22[z])[0]
                            break
                    if self.I == 4:  # Voice Type
                        if len(chosenprob2index) <= index2:
                            index2 = index2 - 1
                            if index2 == -1:
                                index2 = 0
                            value2 = 1
                        path2 = self.g11[chosenprob2index[index2], :]
                    elif self.I == 5:
                        if len(chosenprob2index) <= index2:
                            index2 = index2 - 1
                            if index2 == -1:
                                index2 = 0
                            value2 = 1
                        path2 = self.h11[chosenprob2index[index2], :]
                    elif self.I == 6:
                        if len(chosenprob2index) <= index2:
                            index2 = index2 - 1
                            if index2 == -1:
                                index2 = 0
                            value2 = 1
                        path2 = self.i11[chosenprob2index[index2], :]
                    elif self.I == 7:
                        if len(chosenprob2index) <= index2:
                            index2 = index2 - 1
                            if index2 == -1:
                                index2 = 0
                            value2 = 1
                        path2 = self.j11[chosenprob2index[index2], :]
                    path2size = np.shape(path2)[0]
                    for i in range(0, path2size - 1, 1):
                        if path2[i + 1] != 0:
                            if (1 / self.wt_matx_real1[path2[i] - 1, path2[i + 1] - 1]) - self.min_rate_new < 0:
                                value2 = 0
                                self.chosenprob2 = np.delete(self.chosenprob2, index2, axis=0)
                                chosenprob2index = np.delete(chosenprob2index, index2, axis=0)
                                break
                            elif (1 / self.wt_matx[path2[i] - 1, path2[i + 1] - 1]) - self.min_rate_new < 0:
                                value2 = 0
                                self.chosenprob2 = np.delete(self.chosenprob2, index2, axis=0)
                                chosenprob2index = np.delete(chosenprob2index, index2, axis=0)
                                break
                        else:
                            break
                    if value2 == 1:
                        break
            else:
                check2 = 1
        [s1, s2] = np.shape(self.path_final)
        self.callblock = 1
        noofpaths = 1
        if check1 == 1 or check2 == 1:
            self.callblock = 0
        else:
            if path1size < self.p:
                for j in range(path1size, self.p, 1):
                    path1 = np.append(path1, 0)
            if path2size < self.p:
                for j in range(path2size, self.p, 1):
                    path2 = np.append(path2, 0)
            self.fwdpath = path1
            self.bkwdpath = path2
            for i in range(0, path1size -1, 1):
                if path1[i+1] != 0:
                    value_real1 = 1 / self.wt_matx_real1[path1[i] - 1, path1[i + 1] - 1] - self.min_rate_new
                    self.wt_matx_real1[path1[i] - 1, path1[i + 1] - 1] = 1 / value_real1

                    value_real = 1 / self.wt_matx_real[path1[i] - 1, path1[i + 1] - 1] - self.min_rate_new
                    self.wt_matx_real[path1[i] - 1, path1[i + 1] - 1] = 1 / value_real

                    value = 1 / self.wt_matx[path1[i]-1, path1[i + 1]-1] - self.min_rate_new
                    self.wt_matx[path1[i]-1, path1[i + 1]-1] = 1/value
                else:
                    break
            for i in range(0, path2size -1, 1):
                if path2[i+1] != 0:
                    value_real1 = 1 / self.wt_matx_real1[path2[i] - 1, path2[i + 1] - 1] - self.min_rate_new
                    self.wt_matx_real1[path2[i] - 1, path2[i + 1] - 1] = 1 / value_real1

                    value_real = 1 / self.wt_matx_real[path2[i] - 1, path2[i + 1] - 1] - self.min_rate_new
                    self.wt_matx_real[path2[i] - 1, path2[i + 1] - 1] = 1 / value_real

                    value = 1 / self.wt_matx[path2[i]-1, path2[i + 1]-1] - self.min_rate_new
                    self.wt_matx[path2[i]-1, path2[i + 1]-1] = 1/value
                else:
                    break
            for loop in range(0, s1, 1):
                if self.path_final[loop, 0] == 0:
                    v = [self.flownumber_new, self.flow_type_new,
                         self.userpriority_new, noofpaths,
                         self.min_rate_new]
                    self.path_final[loop, :] = np.concatenate((v, path1))
                    v1 = [self.flownumber_new, self.flow_type_new,
                         self.userpriority_new, noofpaths,
                         self.min_rate_new]
                    self.path_final[loop+1, :] = np.concatenate((v1, path2))
                    break
        self.s_multi = np.append(self.s_multi, self.s_new)
        self.d_multi = np.append(self.d_multi, self.d_new)
        self.flow_type_multi = np.append(self.flow_type_multi, self.flow_type_new)
        self.min_rate_multi = np.append(self.min_rate_multi, self.min_rate_new)
        self.flownumber_multi = np.append(self.flownumber_multi, self.flownumber_new)
        self.userpriority_multi = np.append(self.userpriority_multi, self.userpriority_new)
        self.blockstate_multi = np.append(self.blockstate_multi, self.callblock)
