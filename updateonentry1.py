import numpy as np
from call1 import call1
from call2 import call2
from dijkstra import dijkstra


def findRoute(p, s, d, flow_type, min_rate, flownumber, userpriority, s_new, d_new,
             flow_type_new, min_rate_new, flownumber_new, userpriority_new, path_final, wt_matx,
             wt_matx_real, wt_matx_real1, blockstate, flow_duration, flowarrival_time, connection_type,
             packet_size, packet_datarate, header_size):
    scale = 100.0
    [s1, s2] = np.shape(path_final)
    if flow_type_new == 0:
        if connection_type == 0:  # Two way Voice Calls
            # Calls call1.py
            # Source to destination calculation
            cd = call1(p, s_new, d_new, flow_type_new, min_rate_new, wt_matx,
                    wt_matx_real, wt_matx_real1, connection_type)
            cd.execute()
            path1 = cd.path
            wt_matx = cd.wt_matx
            wt_matx_real = cd.wt_matx_real
            wt_matx_real1 = cd.wt_matx_real1
            # np.savetxt("pathfinal_flowtype0.csv", path1, delimiter=",")
            v = path1 == np.zeros((p))
            if v.all():
                blockstate_new = 0  # Represents blockstate
            else:
                # Destination to source calculation
                cd = call1(p, d_new, s_new, flow_type_new, min_rate_new, wt_matx,
                        wt_matx_real, wt_matx_real1, connection_type)
                cd.execute()
                path2 = cd.path
                wt_matx = cd.wt_matx
                wt_matx_real = cd.wt_matx_real
                wt_matx_real1 = cd.wt_matx_real1
                v = path2 == np.zeros((p))
                if v.all():
                    blockstate_new = 0
                    # Call2
                    # If blocked by destination to source, we invoke Call2
                    cd1 = call2(p, path1, flow_type_new, min_rate_new, wt_matx,
                                wt_matx_real, wt_matx_real1)
                    cd1.execute()
                    wt_matx = cd1.wt_matx
                    wt_matx_real = cd1.wt_matx_real
                    wt_matx_real1 = cd1.wt_matx_real1
                else:
                    blockstate_new = 1
                    # Packets level Variables
                    fwdpath = path1
                    bkwdpath = path2
                    noofpaths = 1
                    for loop in range(0, s1-1, 1):
                        # Not sure about this loop[0] or loop[1]
                        if path_final[loop][0] == 0:
                            # print path_final
                            # print path1
                            if int(flow_duration) * int(packet_datarate / scale / (packet_size - header_size)) < 1:
                                no_of_packets = int(flow_duration) * 1
                            else:
                                no_of_packets = int(flow_duration) * int(packet_datarate / scale / (packet_size - header_size))

                            v = [flownumber_new, flow_type_new,
                                 no_of_packets, connection_type,
                                 min_rate_new, flowarrival_time, flowarrival_time, flow_duration, packet_datarate / scale, 0, no_of_packets, 0, 0, 0, 0]
                            pp = np.concatenate((v, path1))
                            path_final[loop, :] = np.concatenate((pp, [0]))



                            v1 = [flownumber_new, flow_type_new,
                                 no_of_packets, connection_type,
                                 min_rate_new, flowarrival_time, flowarrival_time, flow_duration, packet_datarate / scale, 0, no_of_packets, 0, 0, 0, 0]
                            pp1 = np.concatenate((v, path2))
                            path_final[loop + 1, :] = np.concatenate((pp1, [0]))
                            # np.savetxt("pathfinal1.csv", path_final, delimiter=",")
                            break
        else:  # One Way Video Calls
            # Source to destination calculation
            cd3 = call1(p, s_new, d_new, flow_type_new, min_rate_new, wt_matx,
                    wt_matx_real, wt_matx_real1, connection_type)
            cd3.execute()
            path1 = cd3.path
            wt_matx = cd3.wt_matx
            wt_matx_real = cd3.wt_matx_real
            wt_matx_real1 = cd3.wt_matx_real1
            # np.savetxt("pathfinal_flowtype0.csv", path1, delimiter=",")
            v = path1 == np.zeros((p))
            if v.all():
                blockstate_new = 0  # Represents blockstate
            else:
                blockstate_new = 1
                # Packets level Variables
                fwdpath = path1
                bkwdpath = None  # None is python equivalent for nil
                noofpaths = 1
                for loop in range(0, s1-1, 1):
                    # Not sure about this loop[0] or loop[1]
                    if path_final[loop][0] == 0:
                        # print path_final
                        # print path1
                        if int(flow_duration) * int(packet_datarate / scale / (packet_size - header_size)) < 1:
                            no_of_packets = int(flow_duration) * 1
                        else:
                            no_of_packets = int(flow_duration) * int(packet_datarate / scale / (packet_size - header_size))

                        v = [flownumber_new, flow_type_new,
                            no_of_packets, connection_type,
                            min_rate_new, flowarrival_time, flowarrival_time, flow_duration, packet_datarate/scale, 0, no_of_packets, 0, 0, 0, 0]
                        pp = np.concatenate((v, path1))
                        path_final[loop, :] = np.concatenate((pp, [0]))
                        break
    elif flow_type_new == 1:
        cd = call1(p, s_new, d_new, flow_type_new, min_rate_new, wt_matx,
                   wt_matx_real, wt_matx_real1)
        cd.execute()
        path1 = cd.path
        wt_matx = cd.wt_matx
        wt_matx_real = cd.wt_matx_real
        wt_matx_real1 = cd.wt_matx_real1
        v = path1 == np.zeros((p))
        if v.all():
            blockstate_new = 0
        else:
            blockstate_new = 1
            # Packets level Variables
            fwdpath = path1
            bkwdpath = None
            noofpaths = 1
            for loop in range(0, s1, 1):
                # Not sure about this loop[0] or loop[1]
                if path_final[loop][0] == 0:
                    v = [flownumber_new, flow_type_new,
                         userpriority_new, noofpaths,
                         min_rate_new, flowarrival_time, flowarrival_time, flow_duration, 1]
                    path_final[loop, :] = np.concatenate((v, path1))
                    # np.savetxt("pathfinal2.csv", path_final, delimiter=",")
                    break
    elif flow_type_new == 2:
        cd = call1(p, s_new, d_new, flow_type_new, min_rate_new, wt_matx,
                   wt_matx_real, wt_matx_real1, connection_type)
        cd.execute()
        path1 = cd.path
        # print "PATH12"
        # print path1
        wt_matx = cd.wt_matx
        wt_matx_real = cd.wt_matx_real
        wt_matx_real1 = cd.wt_matx_real1
        v = path1 == np.zeros((p))
        if v.all():
            blockstate_new = 0
        else:
            blockstate_new = 1
            fwdpath = path1
            bkwdpath = None  # None is python equivalent for nil
            noofpaths = 1
            for loop in range(0, s1, 1):
                # print loop
                # Not sure about this loop[0] or loop[1]
                if path_final[loop][0] == 0:
                    # if int(flow_duration * (packet_datarate/scale) / packet_size) < 1:
                    if (flow_duration/(packet_size - header_size)/scale) < 1:
                        file_limit = 1
                    else:
                        file_limit = int(flow_duration/(packet_size - header_size)/scale)
                    if s_new == 3 and d_new == 13:
                        satellite_link = 1
                    else:
                        satellite_link = 0

                    v = [flownumber_new, flow_type_new,
                         file_limit, connection_type,
                         min_rate_new, flowarrival_time, flowarrival_time, flow_duration, packet_datarate/scale, file_limit, 0, 0, satellite_link, 0, 0]
                    pp = np.concatenate((v, path1))
                    path_final[loop, :] = np.concatenate((pp, [0]))
                    break
    else:
        dij = dijkstra(s_new, d_new, wt_matx)
        path1 = dij.execute()
        [path1size1, path1size] = np.shape(path1)
        if path1size < p:
            for loop in range(path1size, p, 1):
                path1[loop] = 0
        blockstate_new = 1
        noofpaths = 1
        for loop in range(0, s1, 1):
            # Not sure about this loop[0] or loop[1]
            if path_final[loop][0] == 0:
                v = [flownumber_new, flow_type_new,
                     userpriority_new, noofpaths,
                     min_rate_new]
                path_final[loop, :] = np.concatenate((v, path1))
                np.savetxt("pathfinal3.csv", path_final, delimiter=",")
                break
    if blockstate_new == 1:
        s = np.append(s, s_new)
        d = np.append(d, d_new)
        flow_type = np.append(flow_type, flow_type_new)
        min_rate = np.append(min_rate, min_rate_new)
        flownumber = np.append(flownumber, flownumber_new)
        userpriority = np.append(userpriority, userpriority_new)
        blockstate = np.append(blockstate, blockstate_new)
    return s, d, flow_type, min_rate, flownumber, userpriority, blockstate, blockstate_new, wt_matx, wt_matx_real, wt_matx_real1, path_final

'''
import numpy as np
import random
import time


class allocaterealupdated1(object):

def __init__(self, p, s, d, flow_type, min_rate, flownumber, userpriority, s_new, d_new,
             flow_type_new, min_rate_new, flownumber_new, userpriority_new, path_final, wt_matx,
             wt_matx_real, wt_matx_real1, blockstate, flow_duration, flowarrival_time, connection_type,
             voice_packet_size, packet_datarate, header_size, wt_matx_real2, probs, I):
    p = p
    s = s
    d = d
    flow_type = flow_type
    min_rate = min_rate
    flownumber = flownumber
    userpriority = userpriority
    s_new = s_new
    d_new = d_new
    flow_type_new = flow_type_new
    min_rate_new = min_rate_new
    flownumber_new = flownumber_new
    userpriority_new = userpriority_new
    path_final = path_final
    wt_matx = wt_matx
    wt_matx_real = wt_matx_real
    wt_matx_real1 = wt_matx_real1
    blockstate = blockstate
    flow_duration = flow_duration
    flowarrival_time = flowarrival_time
    connection_type = connection_type
    packet_size = voice_packet_size
    packet_datarate = packet_datarate
    header_size = header_size
    wt_matx_real2 = wt_matx_real2
    probs = probs
    I = I
    blockstate_new = 1

def execute(self, cpaths):
    scale = 100.0
    [s1, s2] = np.shape(path_final)
    chosenprob = []
    chosenprob1 = probs
    for cho in chosenprob1:
        chosenprob.append(cho[0])
    a11 = cpaths
    chosenpaths = a11
    [a11_row, a11_col] = np.shape(chosenpaths)
    # available_paths = len(chosenprob)
    available_paths = 2
    value1 = 1
    if connection_type == 1:  # Video
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
                    if (1 / wt_matx_real1[int(path[i]) - 1, int(path[i + 1]) - 1]) - min_rate_new < 0:
                        value1 = 0
                        # chosenprob.pop(prob)
                        # chosenpaths.pop(prob)
                        break
                    elif (1 / wt_matx[int(path[i]) - 1, int(path[i + 1]) - 1]) - min_rate_new < 0:
                        value1 = 0
                        # chosenprob.pop(prob)
                        # chosenpaths.pop(prob)
                        break
                else:
                    break
            if value1 == 1:  # Path is valid
                for i in range(0, path1size - 1, 1):
                    if path[i+1] != 0:
                        value_real1 = 1 / wt_matx_real1[int(path[i]) - 1, int(path[i + 1]) - 1] - min_rate_new
                        wt_matx_real1[int(path[i]) - 1, int(path[i + 1]) - 1] = 1 / value_real1

                        value_real = 1 / wt_matx_real[int(path[i]) - 1, int(path[i + 1]) - 1] - min_rate_new
                        wt_matx_real[int(path[i]) - 1, int(path[i + 1]) - 1] = 1 / value_real

                        value = 1 / wt_matx[int(path[i])-1, int(path[i + 1])-1] - min_rate_new
                        wt_matx[int(path[i])-1, int(path[i + 1])-1] = 1/value
                    else:
                        break

                for loop in range(0, s1 - 1, 1):
                    # Not sure about this loop[0] or loop[1]
                    if path_final[loop][0] == 0:
                        # print path_final
                        # print path1
                        if int(packet_datarate / scale / packet_size) < 1:
                            no_of_packets = int(flow_duration) * 1
                        else:
                            no_of_packets = int(flow_duration) * int(packet_datarate / scale / (packet_size - header_size))

                        v = [flownumber_new, flow_type_new,
                             no_of_packets, connection_type,
                             min_rate_new, flowarrival_time, flowarrival_time, flow_duration, packet_datarate / scale, 0, no_of_packets, 0, 0, 0, 0]
                        if path1size < p:
                            for loop1 in range(path1size, p, 1):
                                path = np.concatenate((path, [0]))
                        pp = np.concatenate((v, path))
                        path_final[loop, :] = np.concatenate((pp, [0]))
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
                    if (1 / wt_matx[int(path[i]) - 1, int(path[i + 1]) - 1]) - min_rate_new < 0:
                        value1 = 0
                        # chosenprob.pop(prob)
                        # chosenpaths.pop(prob)
                        break
                else:
                    break
            if value1 == 1:  # Path is valid
                for i in range(0, path1size - 1, 1):
                    if path[i + 1] != 0:
                        value = 1 / wt_matx[int(path[i]) - 1, int(path[i + 1]) - 1] - min_rate_new
                        wt_matx[int(path[i]) - 1, int(path[i + 1]) - 1] = 1 / value
                    else:
                        break

                for loop in range(0, s1, 1):
                    # print loop
                    # Not sure about this loop[0] or loop[1]
                    if path_final[loop][0] == 0:
                        # if int(flow_duration * (packet_datarate/scale) / packet_size) < 1:
                        if int(flow_duration / packet_size / scale) < 1:
                            file_limit = 1
                        else:
                            file_limit = int(flow_duration / (packet_size - header_size) / scale)
                        v = [flownumber_new, flow_type_new,
                             file_limit, connection_type,
                             min_rate_new, flowarrival_time, flowarrival_time, flow_duration, packet_datarate / scale, file_limit, 0, 0, 0, 0, 0]
                        if path1size < p:
                            for loop1 in range(path1size, p, 1):
                                path = np.concatenate((path, [0]))
                        pp = np.concatenate((v, path))
                        path_final[loop, :] = np.concatenate((pp, [0]))
                        break
                break
            else:
                # Blocked. Hence, overflow.
                available_paths -= 1
    if available_paths == 0:
        blockstate_new = 0
    else:
        blockstate_new = 1
        s = np.append(s, s_new)
        d = np.append(d, d_new)
        flow_type = np.append(flow_type, flow_type_new)
        min_rate = np.append(min_rate, min_rate_new)
        flownumber = np.append(flownumber, flownumber_new)
        userpriority = np.append(userpriority, userpriority_new)
        blockstate = np.append(blockstate, blockstate_new)
'''