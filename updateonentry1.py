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
                 wt_matx_real, wt_matx_real1, blockstate, flow_duration, flowarrival_time, connection_type,
                 voice_packet_size, packet_datarate, header_size):
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
        self.flow_duration = flow_duration
        self.flowarrival_time = flowarrival_time
        self.connection_type = connection_type
        self.packet_size = voice_packet_size
        self.packet_datarate = packet_datarate
        self.header_size = header_size

    def execute(self):
        scale = 100.0
        [s1, s2] = np.shape(self.path_final)
        if self.flow_type_new == 0:
            if self.connection_type == 0:  # Two way Voice Calls
                # Calls call1.py
                # Source to destination calculation
                cd = call1(self.p, self.s_new, self.d_new, self.flow_type_new, self.min_rate_new, self.wt_matx,
                        self.wt_matx_real, self.wt_matx_real1, self.connection_type)
                cd.execute()
                path1 = cd.path
                self.wt_matx = cd.wt_matx
                self.wt_matx_real = cd.wt_matx_real
                self.wt_matx_real1 = cd.wt_matx_real1
                # np.savetxt("pathfinal_flowtype0.csv", path1, delimiter=",")
                v = path1 == np.zeros((self.p))
                if v.all():
                    self.blockstate_new = 0  # Represents blockstate
                else:
                    # Destination to source calculation
                    cd = call1(self.p, self.d_new, self.s_new, self.flow_type_new, self.min_rate_new, self.wt_matx,
                            self.wt_matx_real, self.wt_matx_real1, self.connection_type)
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
                        # Packets level Variables
                        self.fwdpath = path1
                        self.bkwdpath = path2
                        noofpaths = 1
                        for loop in range(0, s1-1, 1):
                            # Not sure about this loop[0] or loop[1]
                            if self.path_final[loop][0] == 0:
                                # print self.path_final
                                # print path1
                                if int(self.flow_duration) * int(self.packet_datarate / scale / (self.packet_size - self.header_size)) < 1:
                                    no_of_packets = int(self.flow_duration) * 1
                                else:
                                    no_of_packets = int(self.flow_duration) * int(self.packet_datarate / scale / (self.packet_size - self.header_size))

                                v = [self.flownumber_new, self.flow_type_new,
                                     no_of_packets, self.connection_type,
                                     self.min_rate_new, self.flowarrival_time, self.flowarrival_time, self.flow_duration, self.packet_datarate / scale, 0, no_of_packets, 0, 0, 0, 0]
                                pp = np.concatenate((v, path1))
                                self.path_final[loop, :] = np.concatenate((pp, [0]))



                                v1 = [self.flownumber_new, self.flow_type_new,
                                     no_of_packets, self.connection_type,
                                     self.min_rate_new, self.flowarrival_time, self.flowarrival_time, self.flow_duration, self.packet_datarate / scale, 0, no_of_packets, 0, 0, 0, 0]
                                pp1 = np.concatenate((v, path2))
                                self.path_final[loop + 1, :] = np.concatenate((pp1, [0]))
                                # np.savetxt("pathfinal1.csv", self.path_final, delimiter=",")
                                break
            else:  # One Way Video Calls
                # Source to destination calculation
                cd3 = call1(self.p, self.s_new, self.d_new, self.flow_type_new, self.min_rate_new, self.wt_matx,
                        self.wt_matx_real, self.wt_matx_real1, self.connection_type)
                cd3.execute()
                path1 = cd3.path
                self.wt_matx = cd3.wt_matx
                self.wt_matx_real = cd3.wt_matx_real
                self.wt_matx_real1 = cd3.wt_matx_real1
                # np.savetxt("pathfinal_flowtype0.csv", path1, delimiter=",")
                v = path1 == np.zeros((self.p))
                if v.all():
                    self.blockstate_new = 0  # Represents blockstate
                else:
                    self.blockstate_new = 1
                    # Packets level Variables
                    self.fwdpath = path1
                    self.bkwdpath = None  # None is python equivalent for nil
                    noofpaths = 1
                    for loop in range(0, s1-1, 1):
                        # Not sure about this loop[0] or loop[1]
                        if self.path_final[loop][0] == 0:
                            # print self.path_final
                            # print path1
                            if int(self.flow_duration) * int(self.packet_datarate / scale / (self.packet_size - self.header_size)) < 1:
                                no_of_packets = int(self.flow_duration) * 1
                            else:
                                no_of_packets = int(self.flow_duration) * int(self.packet_datarate / scale / (self.packet_size - self.header_size))

                            v = [self.flownumber_new, self.flow_type_new,
                                no_of_packets, self.connection_type,
                                self.min_rate_new, self.flowarrival_time, self.flowarrival_time, self.flow_duration, self.packet_datarate/scale, 0, no_of_packets, 0, 0, 0, 0]
                            pp = np.concatenate((v, path1))
                            self.path_final[loop, :] = np.concatenate((pp, [0]))
                            break
        elif self.flow_type_new == 1:
            cd = call1(self.p, self.s_new, self.d_new, self.flow_type_new, self.min_rate_new, self.wt_matx,
                       self.wt_matx_real, self.wt_matx_real1)
            cd.execute()
            path1 = cd.path
            self.wt_matx = cd.wt_matx
            self.wt_matx_real = cd.wt_matx_real
            self.wt_matx_real1 = cd.wt_matx_real1
            v = path1 == np.zeros((self.p))
            if v.all():
                self.blockstate_new = 0
            else:
                self.blockstate_new = 1
                # Packets level Variables
                self.fwdpath = path1
                self.bkwdpath = None
                noofpaths = 1
                for loop in range(0, s1, 1):
                    # Not sure about this loop[0] or loop[1]
                    if self.path_final[loop][0] == 0:
                        v = [self.flownumber_new, self.flow_type_new,
                             self.userpriority_new, noofpaths,
                             self.min_rate_new, self.flowarrival_time, self.flowarrival_time, self.flow_duration, 1]
                        self.path_final[loop, :] = np.concatenate((v, path1))
                        # np.savetxt("pathfinal2.csv", self.path_final, delimiter=",")
                        break
        elif self.flow_type_new == 2:
            cd = call1(self.p, self.s_new, self.d_new, self.flow_type_new, self.min_rate_new, self.wt_matx,
                       self.wt_matx_real, self.wt_matx_real1, self.connection_type)
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
                self.fwdpath = path1
                self.bkwdpath = None  # None is python equivalent for nil
                noofpaths = 1
                for loop in range(0, s1, 1):
                    # print loop
                    # Not sure about this loop[0] or loop[1]
                    if self.path_final[loop][0] == 0:
                        # if int(self.flow_duration * (self.packet_datarate/scale) / self.packet_size) < 1:
                        if (self.flow_duration/(self.packet_size - self.header_size)/scale) < 1:
                            file_limit = 1
                        else:
                            file_limit = int(self.flow_duration/(self.packet_size - self.header_size)/scale)
                        if self.s_new == 3 and self.d_new == 13:
                            satellite_link = 1
                        else:
                            satellite_link = 0

                        v = [self.flownumber_new, self.flow_type_new,
                             file_limit, self.connection_type,
                             self.min_rate_new, self.flowarrival_time, self.flowarrival_time, self.flow_duration, self.packet_datarate/scale, file_limit, 0, 0, satellite_link, 0, 0]
                        pp = np.concatenate((v, path1))
                        self.path_final[loop, :] = np.concatenate((pp, [0]))
                        break
        else:
            dij = dijkstra(self.s_new, self.d_new, self.wt_matx)
            path1 = dij.execute()
            [path1size1, path1size] = np.shape(path1)
            if path1size < self.p:
                for loop in range(path1size, self.p, 1):
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

'''
import numpy as np
import random
import time


class allocaterealupdated1(object):

    def __init__(self, p, s, d, flow_type, min_rate, flownumber, userpriority, s_new, d_new,
                 flow_type_new, min_rate_new, flownumber_new, userpriority_new, path_final, wt_matx,
                 wt_matx_real, wt_matx_real1, blockstate, flow_duration, flowarrival_time, connection_type,
                 voice_packet_size, packet_datarate, header_size, wt_matx_real2, probs, I):
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
        self.flow_duration = flow_duration
        self.flowarrival_time = flowarrival_time
        self.connection_type = connection_type
        self.packet_size = voice_packet_size
        self.packet_datarate = packet_datarate
        self.header_size = header_size
        self.wt_matx_real2 = wt_matx_real2
        self.probs = probs
        self.I = I
        self.blockstate_new = 1

    def execute(self, cpaths):
        scale = 100.0
        [s1, s2] = np.shape(self.path_final)
        chosenprob = []
        chosenprob1 = self.probs
        for cho in chosenprob1:
            chosenprob.append(cho[0])
        a11 = cpaths
        chosenpaths = a11
        [a11_row, a11_col] = np.shape(chosenpaths)
        # available_paths = len(chosenprob)
        available_paths = 2
        value1 = 1
        if self.connection_type == 1:  # Video
            sorted_prob = sorted(chosenprob, reverse=True)
            sorted_index = np.argsort(chosenprob)
            weighted_probs = []
            weighted_index = []
            # Obtain the top probabilities
            for kl in range(0, len(sorted_prob), 1):
                if sorted_prob[kl] >= 0.01:
                    weighted_probs.append(sorted_prob[kl])
                    weighted_index.append(sorted_index[kl])
            min_prob = min(weighted_probs)
            choosing_probs = []
            for kp in range(0, len(weighted_probs), 1):
                choosing_probs = np.append(choosing_probs, [weighted_index[kp]] * int(round(weighted_probs[kp] / min_prob)))
            # prob_vector = np.random.choice(a11_row, a11_row, replace=False, p=chosenprob)
            # prob_vector_index = 0
            while available_paths != 0:
                prob_vector = np.random.choice(choosing_probs)
                value1 = 1
                # prob = prob_vector[prob_vector_index]
                # prob_vector_index += 1
                path = chosenpaths[int(prob_vector)]
                path1size = np.shape(path)[0]
                for i in range(0, path1size - 1, 1):
                    if path[i + 1] != 0:
                        if (1 / self.wt_matx_real1[int(path[i]) - 1, int(path[i + 1]) - 1]) - self.min_rate_new < 0:
                            value1 = 0
                            # chosenprob.pop(prob)
                            # chosenpaths.pop(prob)
                            break
                        elif (1 / self.wt_matx[int(path[i]) - 1, int(path[i + 1]) - 1]) - self.min_rate_new < 0:
                            value1 = 0
                            # chosenprob.pop(prob)
                            # chosenpaths.pop(prob)
                            break
                    else:
                        break
                if value1 == 1:  # Path is valid
                    for i in range(0, path1size - 1, 1):
                        if path[i+1] != 0:
                            value_real1 = 1 / self.wt_matx_real1[int(path[i]) - 1, int(path[i + 1]) - 1] - self.min_rate_new
                            self.wt_matx_real1[int(path[i]) - 1, int(path[i + 1]) - 1] = 1 / value_real1

                            value_real = 1 / self.wt_matx_real[int(path[i]) - 1, int(path[i + 1]) - 1] - self.min_rate_new
                            self.wt_matx_real[int(path[i]) - 1, int(path[i + 1]) - 1] = 1 / value_real

                            value = 1 / self.wt_matx[int(path[i])-1, int(path[i + 1])-1] - self.min_rate_new
                            self.wt_matx[int(path[i])-1, int(path[i + 1])-1] = 1/value
                        else:
                            break

                    for loop in range(0, s1 - 1, 1):
                        # Not sure about this loop[0] or loop[1]
                        if self.path_final[loop][0] == 0:
                            # print self.path_final
                            # print path1
                            if int(self.packet_datarate / scale / self.packet_size) < 1:
                                no_of_packets = int(self.flow_duration) * 1
                            else:
                                no_of_packets = int(self.flow_duration) * int(self.packet_datarate / scale / (self.packet_size - self.header_size))

                            v = [self.flownumber_new, self.flow_type_new,
                                 no_of_packets, self.connection_type,
                                 self.min_rate_new, self.flowarrival_time, self.flowarrival_time, self.flow_duration, self.packet_datarate / scale, 0, no_of_packets, 0, 0, 0, 0]
                            if path1size < self.p:
                                for loop1 in range(path1size, self.p, 1):
                                    path = np.concatenate((path, [0]))
                            pp = np.concatenate((v, path))
                            self.path_final[loop, :] = np.concatenate((pp, [0]))
                            break
                    break
                else:
                    # Blocked. Hence, overflow.
                    available_paths -= 1
        else:
            sorted_prob = sorted(chosenprob, reverse=True)
            sorted_index = np.argsort(chosenprob)
            weighted_probs = []
            weighted_index = []
            # Obtain the top probabilities
            for kl in range(0, len(sorted_prob), 1):
                if sorted_prob[kl] >= 0.01:
                    weighted_probs.append(sorted_prob[kl])
                    weighted_index.append(sorted_index[kl])
            min_prob = min(weighted_probs)
            choosing_probs = []
            for kp in range(0, len(weighted_probs), 1):
                choosing_probs = np.append(choosing_probs, [weighted_index[kp]]*int(round(weighted_probs[kp]/min_prob)))

            # prob_vector = np.random.choice(a11_row, a11_row, replace=False, p=chosenprob)

            prob_vector_index = 0
            while available_paths != 0:
                prob_vector = np.random.choice(choosing_probs)
                value1 = 1
                # prob = prob_vector[prob_vector_index]
                # prob_vector_index += 1
                # path = chosenpaths[prob]
                path = chosenpaths[int(prob_vector)]
                path1size = np.shape(path)[0]
                for i in range(0, path1size - 1, 1):
                    if path[i + 1] != 0:
                        if (1 / self.wt_matx[int(path[i]) - 1, int(path[i + 1]) - 1]) - self.min_rate_new < 0:
                            value1 = 0
                            # chosenprob.pop(prob)
                            # chosenpaths.pop(prob)
                            break
                    else:
                        break
                if value1 == 1:  # Path is valid
                    for i in range(0, path1size - 1, 1):
                        if path[i + 1] != 0:
                            value = 1 / self.wt_matx[int(path[i]) - 1, int(path[i + 1]) - 1] - self.min_rate_new
                            self.wt_matx[int(path[i]) - 1, int(path[i + 1]) - 1] = 1 / value
                        else:
                            break

                    for loop in range(0, s1, 1):
                        # print loop
                        # Not sure about this loop[0] or loop[1]
                        if self.path_final[loop][0] == 0:
                            # if int(self.flow_duration * (self.packet_datarate/scale) / self.packet_size) < 1:
                            if int(self.flow_duration / self.packet_size / scale) < 1:
                                file_limit = 1
                            else:
                                file_limit = int(self.flow_duration / (self.packet_size - self.header_size) / scale)
                            v = [self.flownumber_new, self.flow_type_new,
                                 file_limit, self.connection_type,
                                 self.min_rate_new, self.flowarrival_time, self.flowarrival_time, self.flow_duration, self.packet_datarate / scale, file_limit, 0, 0, 0, 0, 0]
                            if path1size < self.p:
                                for loop1 in range(path1size, self.p, 1):
                                    path = np.concatenate((path, [0]))
                            pp = np.concatenate((v, path))
                            self.path_final[loop, :] = np.concatenate((pp, [0]))
                            break
                    break
                else:
                    # Blocked. Hence, overflow.
                    available_paths -= 1
        if available_paths == 0:
            self.blockstate_new = 0
        else:
            self.blockstate_new = 1
            self.s = np.append(self.s, self.s_new)
            self.d = np.append(self.d, self.d_new)
            self.flow_type = np.append(self.flow_type, self.flow_type_new)
            self.min_rate = np.append(self.min_rate, self.min_rate_new)
            self.flownumber = np.append(self.flownumber, self.flownumber_new)
            self.userpriority = np.append(self.userpriority, self.userpriority_new)
            self.blockstate = np.append(self.blockstate, self.blockstate_new)
'''