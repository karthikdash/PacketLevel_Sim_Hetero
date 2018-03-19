import numpy as np
from updateonentry1 import findRoute
from Packets import Packets
import bisect
from debugTools import checkSystemResources, displayStats
from PacketTools import nodesHavePackets, realnodesHavePackets, nonrealnodesHavePackets, dataPacketsAvailable, currentNonrealQueue, appendNonRealQueue, removeFlow, removeReverseFlow, removeFileFlow
from allocaterealupdated1 import allocaterealupdated1
from allocatenonrealupdated1 import allocatenonrealupdated1

# Adapted Dijkstra Packet Simulator for One Way Video Streaming

lamb = 0.009  # Arrival lambda at the source
limit = 20000  # Total number of flows the simulation will execute
start = 50  # Statistics are computes from this flow arrival

# In order to increase the number of flows simulated for a given time,
# we scale down the rate requirements and link capacities.
scale = 100.0

#  +++++++++++ Network Parameters ++++++++++++++++ #

min_arrivaltime = 0  # Minimum of all arrival times of the next packet considering all the current flows
time_service = 0.1  # Current system time


n = 10  # Total Number of Nodes
p = n
noOfNodes = n

packetcounter = 0  # Total number of packets which have reached the destination

# Link connections are defined here. link_src[i] is connected to link_dest[i]
link_src = [1, 1, 2, 2, 3, 4, 4, 5, 5, 5, 6, 7, 8, 9]
link_dest = [2, 8, 3, 9, 4, 5, 10, 6, 7, 10, 7, 8, 10, 10]

link_factor = 1  # Useful for increasing the link capacities

numberOfLinks = len(link_src) * 2  # Becuase we take bidrection links, we reverse each link

caps = [2] * numberOfLinks  # Link rate in Mbps
link_rate_orig = np.multiply(link_factor, caps)

packet_size = 500.00  # Bits
header_size = 48.00  # Bits

link_onprob1 = [0.3] * numberOfLinks

link_errorprob1 = [0.05] * numberOfLinks 

link_onprob2 = [0.3] * numberOfLinks

link_errorprob2 = [0.01] * numberOfLinks

link_onprob3 = [0.2] * numberOfLinks

link_errorprob3 = [0.07] * numberOfLinks

# Source-destination pairs : source1[i]-destination1[i]
source1 = [2, 4, 3, 1]
destination1 = [6, 8, 7, 5]


noSD = len(source1)  # Total number of SD pairs

link_rate = np.multiply((10.0**6/packet_size), link_rate_orig) # Frames per second

pure_link_rate = link_rate_orig

# Data Rate Requirements

# Example for 4 SD pairs [Voice_rate_SD1 Voice_rate_SD2 Voice_rate_SD3 Voice_rate_SD4]
# In this scenario, each SD pair generates voice, video and file connections.
voice_require = [22, 80, 22, 11] # Kbps
video_require = [400, 400, 400, 400] # Kbps
file_require = [300, 400, 300, 300] # Kbps

# Number of individual connections
no_of_voiceConnections = len(voice_require)
no_of_videoConnections = len(video_require)
no_of_fileConnections = len(file_require)

total_commodities =  no_of_voiceConnections + no_of_videoConnections + no_of_fileConnections

data_require = voice_require + video_require + file_require
packet_datarate = np.multiply(data_require, 10.0**3)  #bps
payload = packet_size - header_size
min_rate1 = np.multiply(10.0**3/payload, data_require)  # Frames per second

# Video,Voice and File
connectiontypes = 3

voice_connection_type = 0  # Realtime Commodity
video_connection_type = 0  # Realtime Commodity
file_connection_type = 2  # Non-realtime Commodity

# Defining flow type for each S-D pair
flow_type1 = [voice_connection_type]*no_of_voiceConnections + [video_connection_type]*no_of_videoConnections + [file_connection_type]*no_of_fileConnections
arrivalrate = np.multiply(lamb, np.ones((total_commodities))) # Defining flow type for each S-D pair

voice_duration = 150  # Seconds
video_duration = 150  # Seconds
file_size = 7* (10**6)  # Mb

servicetime = [voice_duration]*no_of_voiceConnections + [video_duration]*no_of_videoConnections + [file_size]*no_of_fileConnections


# Effective Weight Calculation
weight_x = ((np.multiply(np.array(link_onprob1), np.subtract(1, link_errorprob1))) +
            (np.multiply(np.array(link_onprob2), np.subtract(1, link_errorprob2))) +
            (np.multiply(np.array(link_onprob3), np.subtract(1, link_errorprob3))))
weight = np.multiply(weight_x, link_rate)

A = np.zeros((n, n))
m1 = np.shape(link_src)[0]

# ####################IMPORTANT ########################
# Matlab indexing starts from 1 but Python starts from 0

# Effective Capacities of Links
for i in range(0, m1):
    q = link_src[i] - 1
    r = link_dest[i] - 1
    A[q][r] = weight[i]  # Forward link
    A[r][q] = weight[i]  # Reverse link
eff_capacity_matx = A
a = np.delete(eff_capacity_matx, 0, 0)
a = np.delete(a, 0, 1)


# Origianl Capacities of Links
B = np.zeros((n, n))
for i in range(0, m1):
    q = link_src[i] - 1
    r = link_dest[i] - 1
    B[q][r] = pure_link_rate[i]  # Forward link
    B[r][q] = pure_link_rate[i]  # Reverse link
b = np.delete(B, 0, 0)
b = np.delete(B, 0, 1)

# C gives the retransmission probability for packet level
C = np.zeros((n, n))
for i in range(0, m1):
    q = link_src[i] - 1
    r = link_dest[i] - 1
    C[q][r] = weight_x[i]  # Forward link
    C[r][q] = weight_x[i]  # Reverse link
c = np.delete(C, 0, 0)
c = np.delete(c, 0, 1)

# To ignore Division by O warnings. /0 taken as Inf
with np.errstate(divide='ignore', invalid='ignore'):
    # Adapted Dijkstra Weights. This is equivalent to 1/ (u - F)
    wt_matx = np.divide(1, eff_capacity_matx)
    wt_matx_real = np.divide(1, eff_capacity_matx)
    wt_matx_real1 = np.divide(1.33, eff_capacity_matx)

source = []
destination = []

# For rho calculations
orig_matx = wt_matx
orig_real1 = wt_matx_real1
rho_matx_sum = np.zeros((n, n))
old_c = 0
new_c = 0
sum_c = 0


# Debugging

orig_total_matx = np.sum(1/wt_matx)
orig_total_real = np.sum(1/wt_matx_real)
orig_total_real1 = np.sum(1/wt_matx_real1)

# To get Source-Destination Pairs defined by problem statement

for i in range(0, connectiontypes):
    source = np.append(source, source1)
    destination = np.append(destination, destination1)

# Adapted Dijkstra Variable Definition
s = []
d = []
flow_type = []
min_rate = []
flownumber = []
userpriority = []
blockstate = []
userpriority_new = 1
flownumber_new = 0


# Intiializations for tracking
sflow = np.zeros((limit))
dflow = np.zeros((limit))
flowtype = np.zeros((limit))
minrate = np.zeros((limit))
userpriority1 = np.zeros((limit))
blockstate1 = np.zeros((limit))
blockstate1_multi = np.zeros((limit))
blockstate1_block = np.zeros((limit))

# ##
count_algo1 = 0
count_alog2 = 0
count_multi = 0
blockalgo1 = 0
blockalgo2 = 0
blockmulti = 0
countarrival = 0
countdeparture = 0

# ##
path_final = np.zeros((limit, p+16))


count1 = np.zeros((limit))

count1departure = np.zeros((limit))
frac = np.zeros((limit-start))

blockfirstattempt1 = np.zeros((limit-start))
blockfirstattempt = 0

countvoice = 0
countvideo = 0
countnonrealtime = 0

totalvoice = 0
totalvideo = 0
totalnonrealtime = 0

blockedvoice_alog1 = 0
blockedvideo_algo1 = 0
blocekednonrealtime_algo1 = 0

blockedvoice_multi = 0
blockedvideo_multi = 0
blocekednonrealtime_multi = 0

blockedvoice_alog2 = 0
blockedvideo_alog2 = 0
blocekednonrealtime_alog2 = 0

delayvolume = np.zeros((limit))
delayvolume_multi = np.zeros((limit))
avgdelay = np.zeros((limit))
avgdelay_multi = np.zeros((limit))

totalcost = 0
totalcost_block = 0
totalcost_multi = 0
timeprevious = float('inf')
avgcost1 = []
avgcost1_block = []
avgcost1_multi = []

arrivaltime = []
departuretime = []
departuretime1 = []
path1 = []
path2 = []

# ######################### Packet Level Initialisations ##########################################


queue_size = []
queue_time = []
firstarrival_time = 0
firstqueue = 0

# limit = input('Limit:')
t = 0

path = [0, 1, 2, 4]

# noOfNodes = 10
packets0 = []
packets_realtime0 = []

packets_tracker0 = []
packets_tracker1 = []
packets_tracker2 = []
packets_tracker3 = []
packets_tracker4 = []

nodes_nonreal = {}
nodes_real = {}
for k1 in range(0, len(link_src), 1):
    nodes_nonreal[link_src[k1], link_dest[k1]] = [[0, 0, 0, 0, 0, 0, 0, 0], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    nodes_nonreal[link_dest[k1], link_src[k1]] = [[0, 0, 0, 0, 0, 0, 0, 0], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    nodes_real[link_src[k1], link_dest[k1]] = []
    nodes_real[link_dest[k1], link_src[k1]] = []


node_links = {
    1: [2, 8],
    2: [3, 9, 1],
    3: [4, 2],
    4: [5, 10, 3],
    5: [6, 7, 10, 4],
    6: [7, 5],
    7: [8, 5, 6],
    8: [10, 1, 7],
    9: [10, 2],
    10: [4, 5, 8, 9],
}



# Slots trackers for rho calculation based on slots

# Total number of slots for which the slot was active
nodes_slot_used = np.zeros((n, n))
nodes_slot_unused = np.zeros((n, n))
nodes_slot_total = np.zeros((n, n))
nodes_total_real_packets = np.zeros((n, n))
nodes_total_nonreal_packets = np.zeros((n, n))
nodes_slot_queue_real_len = np.zeros((n, n))
nodes_slot_queue_nonreal_len = np.zeros((n, n))

Voice_Mean = 0
Video_Mean = 0
File_Mean = 0
File_Mean_Speed = 0
File_Mean_Speed_e2e = 0

Voice_Mean_Total = 0
Video_Mean_Total = 0
File_Mean_Total = 0

Video_e2e = 0
Video_e2e_Count = 0
Video_e2e1 = 0
Video_e2e_Count1 = 0

Voice_e2e = 0
Voice_e2e_Count = 0

File_e2e = 0
File_e2e_Count = 0

sum_soujorn = 0
number_soujorn = 0

File_Mean_Total_Time_Count_Satellite = 0
File_Mean_Total_Time_Satellite = 0

File_Mean_Total_Time_Count_WO_Satellite = 0
File_Mean_Total_Time_WO_Satellite = 0

File_Mean_Total_Time_Count_All = 0
File_Mean_Total_Time_All = 0



serviceend_time = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
final_packets_tracker = []
timetracker = []
len_tracker = 0
fwdpath = []
bkwdpath = []



# ################### End Of Packet level Simulation variables############

# Exponential Random distribution at mean 1/lambda
# All the arrival times are computed here. So each flowarrival[] time corresponds to particular source[]
flowarrivaltime = np.random.exponential(np.divide(1, arrivalrate))
arrivalratesize = np.shape(arrivalrate)[0]
arrivalratesize1 = 1

c = 0

while(countarrival < limit - 1):
    print countarrival, "countarrival", packetcounter , time_service
    if countarrival > 200:
        displayStats(blockedvoice_alog1, totalvoice, blockedvideo_algo1,
                     totalvideo, blocekednonrealtime_algo1, totalnonrealtime,
                     sum_soujorn, number_soujorn, Video_e2e, Video_e2e_Count,
                     Voice_e2e, Voice_e2e_Count, File_e2e, File_e2e_Count, scale, lamb)

    # We find the minimum get the first arriving flow and hence source node for that corresponding time
    c = flowarrivaltime.min()  # Minimum Value
    I = flowarrivaltime.argmin()  # Index of the Minimum Value
    if countarrival == 0:

        timepreviuos1 = flowarrivaltime[I]  # First Flow arrival time

        # Arrivaltime vector is updated by appending the current flow arrival time which just arrived
        arrivaltime = np.append(arrivaltime, flowarrivaltime[I])

        # Source node of the considered flow
        sflow[countarrival] = source[I]

        # Destination node of the considered flow
        dflow[countarrival] = destination[I]

        # Type of flow of the considered flow
        flowtype[countarrival] = flow_type1[I]

        # Rate of the considered flow
        minrate[countarrival] = min_rate1[I]

        # Priority set to 1
        userpriority1[countarrival] = userpriority_new

        # Flow number for Adapted Dijsktra set to 1 for first flow
        flownumber_new = flownumber_new + 1

        flow_duration = np.random.exponential(servicetime[I])

        ########## ADAPTED DIJSKTRA ROUTING ALGORITHM #########################


        if I <= 3:
            connection_type = 0  # Voice Call
        elif I <= 7:
            connection_type = 1  # Video Call
        elif I <= 11:
            connection_type = 2  # Data Call

        s, d, flow_type, min_rate,\
        flownumber, userpriority, blockstate,\
        blockstate_new, wt_matx, wt_matx_real,\
        wt_matx_real1, path_final = findRoute(p, s, d, flow_type, min_rate, flownumber, userpriority, source[I],
                                        destination[I], flow_type1[I], min_rate1[I], flownumber_new, userpriority_new,
                                        path_final, wt_matx, wt_matx_real, wt_matx_real1, blockstate, flow_duration,
                                        flowarrivaltime[I], connection_type, packet_size, packet_datarate[I],
                                        header_size)

        # For rho calculations
        old_c = 0
        c = flowarrivaltime.min()
        new_c = c

        # Debugging
        checkSystemResources(wt_matx, wt_matx_real, wt_matx_real1, path_final, orig_total_matx, orig_total_real1)


        if blockstate_new == 0:  # If call is blocked by apadted Dijkstra
            count_algo1 = count_algo1 + 1
        blockstate1[countarrival] = blockstate_new


        # Updating the next flowarrivaltime
        flowarrivaltime[I] = flowarrivaltime[I] + np.random.exponential(np.divide(1, arrivalrate[I]))
        count1departure[countarrival] = countdeparture
        countarrival = countarrival + 1

    else:  # When countarrival > 1
        # The packet level simulation happens here. There are two conditions for
        # which new flow gets added to the system
        # 1) min_arrivaltime of the current flows is greater than or equal to
        # the next upcoming flow time (c)
        # 2) The nodes don't have packets i.e all the current flows have
        # finished their service (nodesHavePackets(noOfNodes, node_links, nodes_real, nodes_nonreal) == false)

        # c1 = departuretime1.min()  # Minimum Value
        # I1 = departuretime1.argmin()  # Index of the Minimum Value
        if min_arrivaltime < c and path_final[0][0] != 0:  # and nodesHavePackets(noOfNodes, node_links, nodes_real, nodes_nonreal):
            # Packetization
            kk = 0
            while min_arrivaltime < c and path_final[0][0] != 0:  # and nodesHavePackets(noOfNodes, node_links, nodes_real, nodes_nonreal):
                k = 0
                l = 0
                kk += 1
                while path_final[k][0] != 0:
                    j = 0
                    l += 1
                    if l > 100:
                        print "asd"
                    if path_final[0][0] != 0:
                        min_arrivaltime = float('inf')
                    po = 0
                    while path_final[j][0] != 0:
                        if path_final[j][6] < min_arrivaltime and path_final[j][3] != 2:
                            # if path_final[k][5] + path_final[k][7] < time_service:
                            min_arrivaltime = path_final[j][6]
                        j += 1
                        po = po +1
                        if po > 100:
                            print po
                    if float('inf') == min_arrivaltime:
                        min_arrivaltime = time_service
                    if countarrival == 1:
                        time_service = min_arrivaltime
                    packet_check = True
                    # if min_arrivaltime <= time_service + packet_size / 20000:
                    if True:
                        #if path_final[k][3] == 0 and (path_final[k][6] - float(1.0 / ((path_final[k][8]) / packet_size))) <= (time_service + packet_size / 20000):
                        if path_final[k][3] == 0 and (path_final[k][6] <= time_service + packet_size / 20000):
                            nodes_real[(int(path_final[k][15]), int(path_final[k][16]))].append(Packets(
                                path_final[k][6],
                                path_final[k][6],
                                path_final[k][7], 0, path_final[k][15:25].tolist() + [0], path_final[k][0],
                                path_final[k][2], True, path_final[k][8], 0, 0))
                            if countarrival == 1:
                                time_service = path_final[k][6]
                            path_final[k][6] = path_final[k][6] + float(1.0 / ((path_final[k][8]) / packet_size))

                            k += 1

                            nodes_real[(int(path_final[k][15]), int(path_final[k][16]))].append(Packets(
                                path_final[k][6],
                                path_final[k][6],
                                path_final[k][7], 0, path_final[k][15:25].tolist() + [0], path_final[k][0],
                                path_final[k][2], False, path_final[k][8], 0, 0))
                            path_final[k][6] = path_final[k][6] + float(1.0 / ((path_final[k][8]) / packet_size))
                            k += 1
                        #elif path_final[k][3] == 1 and (path_final[k][6] - float(1.0 / ((path_final[k][8]) / packet_size))) <= (time_service + packet_size / 20000):  # Video Calls
                        elif path_final[k][3] == 1 and (path_final[k][6] <= time_service + packet_size / 20000):
                            nodes_real[(int(path_final[k][15]), int(path_final[k][16]))].append(Packets(
                                path_final[k][6],
                                path_final[k][6],
                                path_final[k][7], 1, path_final[k][15:25].tolist() + [0], path_final[k][0],
                                path_final[k][2], True, path_final[k][8], 0, 0))
                            # nodes_total_real_packets[int(path_final[k][11])-1][int(path_final[k][12])-1] += 1
                            if countarrival == 1:
                                time_service = path_final[k][6]
                            path_final[k][6] = path_final[k][6] + float(1.0 / ((path_final[k][8]) / packet_size))
                            path_final[k][10] -= 1
                            k += 1
                        elif path_final[k][3] == 2:  # Data calls
                            if path_final[k][14] == 0:
                                if path_final[k][2] == 0:
                                    path_final[k][2] = 1

                                path_final[k][10] += 1
                                for ii in range(1, len(nodes_nonreal[(int(path_final[k][15]), int(path_final[k][16]))])):
                                    if len(nodes_nonreal[(int(path_final[k][15]), int(path_final[k][16]))][ii]) == 0:
                                        for jj in range(0, int(path_final[k][2])):
                                            nodes_nonreal[(int(path_final[k][15]), int(path_final[k][16]))][ii].append(Packets(
                                                path_final[k][6],
                                                path_final[k][6],
                                                path_final[k][7], 2, path_final[k][15:25].tolist(), path_final[k][0],
                                                path_final[k][2], True, path_final[k][8] / 100.0, 0, 0))
                                        path_final[k][14] = 1  # Complete flow packets have been added to the queue
                                        break
                                # nodes_total_nonreal_packets[int(path_final[k][11])-1][int(path_final[k][12])-1] += 1
                            # path_final[k][6] = path_final[k][6] + float(
                            #     1.0 / ((path_final[k][8]) / packet_size))
                            k += 1
                            if countarrival == 1:
                                time_service = path_final[0][6]
                                # min_arrivaltime = time_service
                        else:
                            k += 1
                    elif nodesHavePackets(noOfNodes, node_links, nodes_real, nodes_nonreal) == False:
                        time_service = time_service + packet_size / 20000
                        for node_no in range(1, noOfNodes + 1, 1):
                            nodes_slot_unused[node_no-1][node_links[node_no][next_nodeno]-1] += 1
                            nodes_slot_total[node_no-1][node_links[node_no][next_nodeno]-1] += 1
                    else:
                        break
                else:
                    j = 0
                    if path_final[0][0] != 0:
                        min_arrivaltime = float('inf')
                    while path_final[j][0] != 0:
                        if path_final[j][6] < min_arrivaltime and path_final[j][3] != 2:
                            # if path_final[j][5] + path_final[j][7] < time_service:
                            min_arrivaltime = path_final[j][6]
                        j += 1
                    if float('inf') == min_arrivaltime:
                        min_arrivaltime = time_service
                # if time_service < min_arrivaltime:
                    # time_service = min_arrivaltime
                #while (time_service) <= min_arrivaltime and (time_service) <= c:  # Can be set true here.
                # NodeServicing

                while path_final[j][0] != 0:
                    if path_final[j][6] < min_arrivaltime and path_final[j][3] != 2:
                        # if path_final[j][5] + path_final[j][7] < time_service:
                        min_arrivaltime = path_final[j][6]
                    j += 1
                while min_arrivaltime <= c and time_service <= min_arrivaltime:
                    packet_check = False
                    if nodesHavePackets(noOfNodes, node_links, nodes_real, nodes_nonreal) == False:
                        time_service = time_service + packet_size / 80000
                        for node_no in range(1, noOfNodes + 1, 1):
                            for next_nodeno in range(0, len(node_links[node_no]), 1):
                                nodes_slot_unused[node_no-1][node_links[node_no][next_nodeno]-1] += 1
                                nodes_slot_total[node_no-1][node_links[node_no][next_nodeno]-1] += 1
                        break
                    for i in range(0, 3, 1):
                        if time_service <= min_arrivaltime:
                            for node_no in range(1, noOfNodes + 1, 1):
                                for next_nodeno in range(0, len(node_links[node_no]), 1):
                                    if len(nodes_real[(node_no, node_links[node_no][next_nodeno])]) > 0 and len(nodes_nonreal[(node_no, node_links[node_no][next_nodeno])]) > 0:
                                        nodes_slot_unused[node_no-1][node_links[node_no][next_nodeno]-1] += 1
                                        nodes_slot_total[node_no-1][node_links[node_no][next_nodeno]-1] += 1
                                    if len(nodes_real[(node_no, node_links[node_no][next_nodeno])]) > 0:
                                        nodes_slot_used[node_no-1][node_links[node_no][next_nodeno]-1] += 1
                                        nodes_slot_total[node_no-1][node_links[node_no][next_nodeno]-1] += 1

                                        s_link = int(nodes_real[(node_no, node_links[node_no][next_nodeno])][0].path[0])
                                        d_link = int(nodes_real[(node_no, node_links[node_no][next_nodeno])][0].path[1])
                                        #if nodes_real[(node_no, node_links[node_no][next_nodeno])][0].arrival_time <= time_service and B[s_link - 1][d_link - 1] == 8:
                                        if B[s_link - 1][d_link - 1] == 8:
                                            if len(nodes_real[(node_no, node_links[node_no][next_nodeno])]) == 0:
                                                continue  # Continue checking other nodes for servicable packets
                                            s_link = int(nodes_real[(node_no, node_links[node_no][next_nodeno])][0].path[0])
                                            d_link = int(nodes_real[(node_no, node_links[node_no][next_nodeno])][0].path[1])
                                            link_retransmit_prob = np.random.choice(np.arange(0, 2), p=[1 - C[s_link - 1][d_link - 1], C[s_link - 1][d_link - 1]])
                                            # link_retransmit_prob = 1
                                            for packetno in range(0, len(nodes_real[(node_no, node_links[node_no][next_nodeno])]), 1):
                                                nodes_real[(node_no, node_links[node_no][next_nodeno])][packetno].addSlotDelay(packet_size / 80000)

                                            nodes_slot_queue_real_len[node_no-1][node_links[node_no][next_nodeno]-1] += len(nodes_real[(node_no, node_links[node_no][next_nodeno])])
                                            nodes_real[(node_no, node_links[node_no][next_nodeno])][0].service(
                                                max(nodes_real[(node_no, node_links[node_no][next_nodeno])][0].arrival_time, time_service),
                                                B[s_link - 1][d_link - 1], False, link_retransmit_prob,
                                                packet_size / 80000)

                                            if link_retransmit_prob == 1:
                                                # Appending to the next node receiving Queue
                                                if nodes_real[(node_no, node_links[node_no][next_nodeno])][0].d_new == 99:
                                                    cur_connection_type = nodes_real[(node_no, node_links[node_no][next_nodeno])][0].flow_tag
                                                    if nodes_real[(node_no, node_links[node_no][next_nodeno])][0].flow_tag == 0:
                                                        ################### Decrementing the packet count from path_final ############
                                                        if nodes_real[(node_no, node_links[node_no][next_nodeno])][0].direction == True:
                                                            # path_final[nodes_real[(node_no, node_links[node_no][next_nodeno])][0].flownumber ][2] -= 1
                                                            path_final, nodes_real, node_no, node_links, next_nodeno, time_service, Voice_e2e, Voice_e2e_Count, Video_e2e, Video_e2e_Count, File_e2e, File_e2e_Count, p, s, d, flow_type, min_rate, flownumber, userpriority, wt_matx, wt_matx_real, wt_matx_real1, blockstate, orig_total_matx, orig_total_real1, connection_type = \
                                                                removeFlow(path_final, nodes_real, node_no, node_links, next_nodeno, time_service, Voice_e2e, Voice_e2e_Count,
                                                                       Video_e2e, Video_e2e_Count, File_e2e, File_e2e_Count, p, s, d, flow_type, min_rate, flownumber, userpriority,
                                                                       wt_matx, wt_matx_real, wt_matx_real1, blockstate, orig_total_matx, orig_total_real1, cur_connection_type)
                                                        else:
                                                            path_final, nodes_real, node_no, node_links, next_nodeno, time_service, Voice_e2e, Voice_e2e_Count, Video_e2e, Video_e2e_Count, File_e2e, File_e2e_Count, p, s, d, flow_type, min_rate, flownumber, userpriority, wt_matx, wt_matx_real, wt_matx_real1, blockstate, orig_total_matx, orig_total_real1, connection_type = \
                                                                removeReverseFlow(path_final, nodes_real, node_no, node_links, next_nodeno, time_service, Voice_e2e, Voice_e2e_Count,
                                                                              Video_e2e, Video_e2e_Count, File_e2e, File_e2e_Count, p, s, d, flow_type, min_rate, flownumber, userpriority,
                                                                              wt_matx, wt_matx_real, wt_matx_real1, blockstate, orig_total_matx, orig_total_real1, cur_connection_type)
                                                    else:
                                                        ################### Decrementing the packet count from path_final ############
                                                        if nodes_real[(node_no, node_links[node_no][next_nodeno])][0].direction == True:
                                                            path_final, nodes_real, node_no, node_links, next_nodeno, time_service, Voice_e2e, Voice_e2e_Count, Video_e2e, Video_e2e_Count, File_e2e, File_e2e_Count, p, s, d, flow_type, min_rate, flownumber, userpriority, wt_matx, wt_matx_real, wt_matx_real1, blockstate, orig_total_matx, orig_total_real1, connection_type = \
                                                                removeFlow(path_final, nodes_real, node_no, node_links, next_nodeno, time_service, Voice_e2e, Voice_e2e_Count,
                                                                       Video_e2e, Video_e2e_Count, File_e2e, File_e2e_Count, p, s, d, flow_type, min_rate, flownumber, userpriority,
                                                                       wt_matx, wt_matx_real, wt_matx_real1, blockstate, orig_total_matx, orig_total_real1, cur_connection_type)
                                                        else:
                                                            path_final, nodes_real, node_no, node_links, next_nodeno, time_service, Voice_e2e, Voice_e2e_Count, Video_e2e, Video_e2e_Count, File_e2e, File_e2e_Count, p, s, d, flow_type, min_rate, flownumber, userpriority, wt_matx, wt_matx_real, wt_matx_real1, blockstate, orig_total_matx, orig_total_real1, connection_type = \
                                                                removeReverseFlow(path_final, nodes_real, node_no, node_links, next_nodeno, time_service, Voice_e2e, Voice_e2e_Count,
                                                                              Video_e2e, Video_e2e_Count, File_e2e, File_e2e_Count, p, s, d, flow_type, min_rate, flownumber, userpriority,
                                                                              wt_matx, wt_matx_real, wt_matx_real1, blockstate, orig_total_matx, orig_total_real1, cur_connection_type)

                                                else:
                                                    # nodes_real[str(nodes_real[(node_no, node_links[node_no][next_nodeno])][0].d_new)].append(nodes_real[(node_no, node_links[node_no][next_nodeno])][0])
                                                    # nodes_real[str(nodes_real[(node_no, node_links[node_no][next_nodeno])][0].d_new)].append(Packets(nodes_real[(node_no, node_links[node_no][next_nodeno])][0].initial_arrival_time, nodes_real[(node_no, node_links[node_no][next_nodeno])][0].service_end_time, nodes_real[(node_no, node_links[node_no][next_nodeno])][0].flow_duration, nodes_real[(node_no, node_links[node_no][next_nodeno])][0].flow_tag, nodes_real[(node_no, node_links[node_no][next_nodeno])][0].path, nodes_real[(node_no, node_links[node_no][next_nodeno])][0].flownumber))
                                                    # nodes_total_real_packets[nodes_real[(node_no, node_links[node_no][next_nodeno])][0].d_new-1][nodes_real[(node_no, node_links[node_no][next_nodeno])][0].path[1]-1] += 1
                                                    bisect.insort_left(nodes_real[(nodes_real[(node_no, node_links[node_no][next_nodeno])][0].d_new, nodes_real[(node_no, node_links[node_no][next_nodeno])][0].path[1])],
                                                                       Packets(nodes_real[(node_no, node_links[node_no][next_nodeno])][0].initial_arrival_time,
                                                                               nodes_real[(node_no, node_links[node_no][next_nodeno])][0].service_end_time,
                                                                               nodes_real[(node_no, node_links[node_no][next_nodeno])][0].flow_duration,
                                                                               nodes_real[(node_no, node_links[node_no][next_nodeno])][0].flow_tag,
                                                                               nodes_real[(node_no, node_links[node_no][next_nodeno])][0].path,
                                                                               nodes_real[(node_no, node_links[node_no][next_nodeno])][0].flownumber,
                                                                               nodes_real[(node_no, node_links[node_no][next_nodeno])][0].noofpackets,
                                                                               nodes_real[(node_no, node_links[node_no][next_nodeno])][0].direction,
                                                                               nodes_real[(node_no, node_links[node_no][next_nodeno])][0].node_service_rate,
                                                                               nodes_real[(node_no, node_links[node_no][next_nodeno])][0].total_slot_time,
                                                                               nodes_real[(node_no, node_links[node_no][next_nodeno])][0].total_slots))
                                                nodes_real[(node_no, node_links[node_no][next_nodeno])].pop(0)
                                    else:
                                        # for i in range(0, int(node_service_rate/packet_size) - 1, 1):
                                        if dataPacketsAvailable(node_no, node_links[node_no][next_nodeno], nodes_nonreal):
                                            current_nr_index = 0
                                            current_nr_index = currentNonrealQueue(node_no, node_links[node_no][next_nodeno], nodes_nonreal)
                                            nodes_slot_used[node_no-1][node_links[node_no][next_nodeno]-1] += 1
                                            nodes_slot_total[node_no-1][node_links[node_no][next_nodeno]-1] += 1

                                            #if nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][0].arrival_time <= time_service and B[s_link - 1][d_link - 1] == 8:
                                            s_link = int(nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].path[0])
                                            d_link = int(nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].path[1])
                                            if B[s_link - 1][d_link - 1] == 8:
                                                if nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].flow_tag == 2:

                                                    # Servicing for each individual node
                                                    if len(nodes_nonreal[(node_no, node_links[node_no][next_nodeno])]) == 0:
                                                        continue  # Continue for other servicable nodes

                                                    if serviceend_time[node_no] == 0:
                                                        initial_service_end = 1
                                                    s_link = int(nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].path[0])
                                                    d_link = int(nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].path[1])
                                                    link_retransmit_prob = np.random.choice(np.arange(0, 2), p=[1 - C[s_link - 1][d_link - 1], C[s_link - 1][d_link - 1]])

                                                    if node_no == 8 and node_links[node_no][next_nodeno] == 13:
                                                        if nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].arrival_time > time_service:
                                                            link_retransmit_prob = 0
                                                    if B[s_link - 1][d_link - 1] == 0:
                                                        print "Inf"

                                                    nodes_slot_queue_nonreal_len[node_no-1][node_links[node_no][next_nodeno]-1] += len(nodes_nonreal[(node_no, node_links[node_no][next_nodeno])])
                                                    nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].service(
                                                        max(nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].arrival_time, time_service), B[s_link - 1][d_link - 1],
                                                        False, link_retransmit_prob, packet_size / (2000000 / scale))
                                                    # Appending to the next node receiving Queue
                                                    if link_retransmit_prob == 1:
                                                        if nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].d_new == 99:  # If packet reached destination we add to the end-to-end final tracker
                                                            ################### Decrementing the packet count from path_final ############
                                                            k = 0
                                                            if nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].direction == True:
                                                                path_final, nodes_nonreal, node_no, node_links, next_nodeno, time_service, File_e2e, sum_soujorn, number_soujorn, File_e2e_Count, p, s, d, flow_type, min_rate, flownumber, userpriority, current_nr_index, wt_matx, wt_matx_real, wt_matx_real1, blockstate, orig_total_matx, orig_total_real1 = \
                                                                    removeFileFlow(path_final, nodes_nonreal, node_no, node_links, next_nodeno, time_service, File_e2e, sum_soujorn, number_soujorn,
                                                                               File_e2e_Count, p, s, d, flow_type, min_rate, flownumber, userpriority, current_nr_index,
                                                                               wt_matx, wt_matx_real, wt_matx_real1, blockstate, orig_total_matx, orig_total_real1)
                                                        else:
                                                            nodes_nonreal = appendNonRealQueue(node_no, node_links[node_no][next_nodeno],current_nr_index, current_nr_index, nodes_nonreal)
                                                        nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index].pop(0)
                            time_service = time_service + packet_size / 80000
                    if time_service <= min_arrivaltime:
                        for node_no in range(1, noOfNodes + 1, 1):
                            for next_nodeno in range(0, len(node_links[node_no]), 1):
                                if len(nodes_real[(node_no, node_links[node_no][next_nodeno])]) > 0:
                                    nodes_slot_used[node_no-1][node_links[node_no][next_nodeno]-1] += 1
                                    nodes_slot_total[node_no-1][node_links[node_no][next_nodeno]-1] += 1
                                    # if nodes_real[(node_no, node_links[node_no][next_nodeno])][0].arrival_time >= max(serviceend_time):
                                    arrivaltimes = []

                                    #if nodes_real[(node_no, node_links[node_no][next_nodeno])][0].arrival_time <= time_service:
                                    if True:
                                        # initial_service_end = serviceend_time[node_no]
                                        # if serviceend_time[node_no] == 0:
                                        #     initial_service_end = 1
                                        # Each node gets serviced here
                                        if len(nodes_real[(node_no, node_links[node_no][next_nodeno])]) == 0:
                                            continue  # Continue checking other nodes for servicable packets
                                        s_link = int(nodes_real[(node_no, node_links[node_no][next_nodeno])][0].path[0])
                                        d_link = int(nodes_real[(node_no, node_links[node_no][next_nodeno])][0].path[1])
                                        link_retransmit_prob = np.random.choice(np.arange(0, 2), p=[1 - C[s_link - 1][d_link - 1], C[s_link - 1][d_link - 1]])
                                        # link_retransmit_prob = 1

                                        for packetno in range(0, len(nodes_real[(node_no, node_links[node_no][next_nodeno])]), 1):
                                            nodes_real[(node_no, node_links[node_no][next_nodeno])][packetno].addSlotDelay(packet_size / 20000)

                                        nodes_slot_queue_real_len[node_no-1][node_links[node_no][next_nodeno]-1] += len(nodes_real[(node_no, node_links[node_no][next_nodeno])])

                                        nodes_real[(node_no, node_links[node_no][next_nodeno])][0].service(max(nodes_real[(node_no, node_links[node_no][next_nodeno])][0].arrival_time, time_service),B[s_link - 1][d_link - 1], False, link_retransmit_prob, packet_size/20000)
                                        # Appending to the serving Queue
                                        # serviceend_time[node_no] = nodes_real[(node_no, node_links[node_no][next_nodeno])][0].service_end_time
                                        # packets_tracker[str(node_no)].append(nodes_real[(node_no, node_links[node_no][next_nodeno])][0])
                                        if link_retransmit_prob == 1:

                                                # if Voice_Mean_Time[node_no] > 1:
                                                    # print "no"
                                            # Appending to the next node receiving Queue
                                            if nodes_real[(node_no, node_links[node_no][next_nodeno])][0].d_new == 99:
                                                cur_connection_type = nodes_real[(node_no, node_links[node_no][next_nodeno])][0].flow_tag
                                                if nodes_real[(node_no, node_links[node_no][next_nodeno])][0].flow_tag == 0:
                                                    if nodes_real[(node_no, node_links[node_no][next_nodeno])][0].direction == True:
                                                        path_final, nodes_real, node_no, node_links, next_nodeno, time_service, Voice_e2e, Voice_e2e_Count, Video_e2e, Video_e2e_Count, File_e2e, File_e2e_Count, p, s, d, flow_type, min_rate, flownumber, userpriority, wt_matx, wt_matx_real, wt_matx_real1, blockstate, orig_total_matx, orig_total_real1, connection_type = \
                                                            removeFlow(path_final, nodes_real, node_no, node_links, next_nodeno, time_service, Voice_e2e, Voice_e2e_Count,
                                                                   Video_e2e, Video_e2e_Count, File_e2e, File_e2e_Count, p, s, d, flow_type, min_rate, flownumber, userpriority,
                                                                   wt_matx, wt_matx_real, wt_matx_real1, blockstate, orig_total_matx, orig_total_real1, cur_connection_type)
                                                    else:
                                                        path_final, nodes_real, node_no, node_links, next_nodeno, time_service, Voice_e2e, Voice_e2e_Count, Video_e2e, Video_e2e_Count, File_e2e, File_e2e_Count, p, s, d, flow_type, min_rate, flownumber, userpriority, wt_matx, wt_matx_real, wt_matx_real1, blockstate, orig_total_matx, orig_total_real1, connection_type = \
                                                            removeReverseFlow(path_final, nodes_real, node_no, node_links, next_nodeno, time_service, Voice_e2e, Voice_e2e_Count,
                                                                          Video_e2e, Video_e2e_Count, File_e2e, File_e2e_Count, p, s, d, flow_type, min_rate, flownumber, userpriority,
                                                                          wt_matx, wt_matx_real, wt_matx_real1, blockstate, orig_total_matx, orig_total_real1, cur_connection_type)
                                                else:
                                                    ################### Decrementing the packet count from path_final ############
                                                    if nodes_real[(node_no, node_links[node_no][next_nodeno])][0].direction == True:
                                                        path_final, nodes_real, node_no, node_links, next_nodeno, time_service, Voice_e2e, Voice_e2e_Count, Video_e2e, Video_e2e_Count, File_e2e, File_e2e_Count, p, s, d, flow_type, min_rate, flownumber, userpriority, wt_matx, wt_matx_real, wt_matx_real1, blockstate, orig_total_matx, orig_total_real1, connection_type = \
                                                            removeFlow(path_final, nodes_real, node_no, node_links, next_nodeno, time_service, Voice_e2e, Voice_e2e_Count,
                                                                   Video_e2e, Video_e2e_Count, File_e2e, File_e2e_Count, p, s, d, flow_type, min_rate, flownumber, userpriority,
                                                                   wt_matx, wt_matx_real, wt_matx_real1, blockstate, orig_total_matx, orig_total_real1, cur_connection_type)
                                                    else:
                                                        path_final, nodes_real, node_no, node_links, next_nodeno, time_service, Voice_e2e, Voice_e2e_Count, Video_e2e, Video_e2e_Count, File_e2e, File_e2e_Count, p, s, d, flow_type, min_rate, flownumber, userpriority, wt_matx, wt_matx_real, wt_matx_real1, blockstate, orig_total_matx, orig_total_real1, connection_type = \
                                                            removeReverseFlow(path_final, nodes_real, node_no, node_links, next_nodeno, time_service, Voice_e2e, Voice_e2e_Count,
                                                                          Video_e2e, Video_e2e_Count, File_e2e, File_e2e_Count, p, s, d, flow_type, min_rate, flownumber, userpriority,
                                                                          wt_matx, wt_matx_real, wt_matx_real1, blockstate, orig_total_matx, orig_total_real1, cur_connection_type)
                                            else:
                                                bisect.insort_left(nodes_real[(nodes_real[(node_no, node_links[node_no][next_nodeno])][0].d_new, nodes_real[(node_no, node_links[node_no][next_nodeno])][0].path[1])],
                                                                   Packets(nodes_real[(node_no, node_links[node_no][next_nodeno])][0].initial_arrival_time,
                                                                           nodes_real[(node_no, node_links[node_no][next_nodeno])][0].service_end_time,
                                                                           nodes_real[(node_no, node_links[node_no][next_nodeno])][0].flow_duration,
                                                                           nodes_real[(node_no, node_links[node_no][next_nodeno])][0].flow_tag,
                                                                           nodes_real[(node_no, node_links[node_no][next_nodeno])][0].path,
                                                                           nodes_real[(node_no, node_links[node_no][next_nodeno])][0].flownumber,
                                                                           nodes_real[(node_no, node_links[node_no][next_nodeno])][0].noofpackets,
                                                                           nodes_real[(node_no, node_links[node_no][next_nodeno])][0].direction,
                                                                           nodes_real[(node_no, node_links[node_no][next_nodeno])][0].node_service_rate,
                                                                           nodes_real[(node_no, node_links[node_no][next_nodeno])][0].total_slot_time,
                                                                           nodes_real[(node_no, node_links[node_no][next_nodeno])][0].total_slots))
                                            nodes_real[(node_no, node_links[node_no][next_nodeno])].pop(0)
                                else:
                                    # for i in range(0, int(node_service_rate/packet_size) - 1, 1):
                                    if dataPacketsAvailable(node_no, node_links[node_no][next_nodeno], nodes_nonreal):
                                        current_nr_index = 0
                                        current_nr_index = currentNonrealQueue(node_no, node_links[node_no][next_nodeno], nodes_nonreal)
                                        nodes_slot_used[node_no-1][node_links[node_no][next_nodeno]-1] += 1
                                        nodes_slot_total[node_no-1][node_links[node_no][next_nodeno]-1] += 1
                                        arrivaltimes = []

                                        #if nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][0].arrival_time <= time_service:
                                        if True:
                                            if nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].flow_tag == 2:
                                                # initial_service_end = serviceend_time[node_no]
                                                # if serviceend_time[node_no] == 0:
                                                #     initial_service_end = 1
                                                # Servicing for each individual node
                                                if len(nodes_nonreal[(node_no, node_links[node_no][next_nodeno])]) == 0:
                                                    continue  # Continue for other servicable nodes

                                                # if serviceend_tim///////////e[node_no] == 0:
                                                #     initial_service_end = 1
                                                if len(nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index]) == 0:
                                                    print "trouble"
                                                s_link = int(nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].path[0])
                                                d_link = int(nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].path[1])
                                                link_retransmit_prob = np.random.choice(np.arange(0, 2), p=[
                                                    1 - C[s_link - 1][d_link - 1], C[s_link - 1][d_link - 1]])

                                                if node_no == 8 and node_links[node_no][next_nodeno] == 13:
                                                    if nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].arrival_time > time_service:
                                                        link_retransmit_prob = 0

                                                if B[s_link - 1][d_link - 1] == 0:
                                                    print "Inf"
                                                # for packetno in range(0, len(nodes_nonreal[(node_no, node_links[node_no][next_nodeno])]), 1):
                                                #     nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][packetno].addSlotDelay(packet_size / 20000)

                                                nodes_slot_queue_nonreal_len[node_no-1][node_links[node_no][next_nodeno]-1] += len(nodes_nonreal[(node_no, node_links[node_no][next_nodeno])])

                                                nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].service(max(nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].arrival_time, time_service), B[s_link - 1][d_link - 1], False, link_retransmit_prob, packet_size/20000)

                                                # Appending to the next node receiving Queue
                                                if link_retransmit_prob == 1:
                                                    if nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].d_new == 99:  # If packet reached destination we add to the end-to-end final tracker
                                                        ################### Decrementing the packet count from path_final ############
                                                        if nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].direction == True:
                                                            path_final, nodes_nonreal, node_no, node_links, next_nodeno, time_service, File_e2e, sum_soujorn, number_soujorn, File_e2e_Count, p, s, d, flow_type, min_rate, flownumber, userpriority, current_nr_index, wt_matx, wt_matx_real, wt_matx_real1, blockstate, orig_total_matx, orig_total_real1 = \
                                                                removeFileFlow(path_final, nodes_nonreal, node_no, node_links, next_nodeno, time_service, File_e2e, sum_soujorn, number_soujorn,
                                                                          File_e2e_Count, p, s, d, flow_type, min_rate, flownumber, userpriority, current_nr_index,
                                                                          wt_matx, wt_matx_real, wt_matx_real1, blockstate, orig_total_matx, orig_total_real1)
                                                    else:
                                                        nodes_nonreal = appendNonRealQueue(node_no, node_links[node_no][next_nodeno], current_nr_index, current_nr_index, nodes_nonreal)
                                                    nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index].pop(0)
                        time_service = time_service + packet_size / 80000
        else:  # Either new flow has arrived or all packets have been served.
            timecurrent = c  # Current Time = Arrival Time
            # print departuretime1, "departuretime1"
            # print flow_type, "flowtype"
            # Initilisations for routing

            # Arrivaltime vector is updated by appending the current flow arrival time which just arrived
            arrivaltime = np.append(arrivaltime, flowarrivaltime[I])
            # <PDQ> Time of departure udpated for the flow that as arrived
            timeofdeparture = c + np.random.exponential(servicetime[I])
            # timeofdeparture = c + 150
            # DepartureTime vector is updated by appending the current flow departure time which just arrived
            departuretime = np.append(departuretime, timeofdeparture)
            departuretime1 = np.append(departuretime1, timeofdeparture)
            # New Arrival time computation for the next flow
            # flowarrivaltime[I] = flowarrivaltime[I] + np.random.exponential(np.divide(1, arrivalrate[I]))


            # flowarrivaltime[I] = flowarrivaltime[I] + 1000
            # Source node of the considered flow
            sflow[countarrival] = source[I]
            # Destination node of the considered flow
            dflow[countarrival] = destination[I]
            # Type of flow of the considered flow
            flowtype[countarrival] = flow_type1[I]
            # Rate of the considered flow
            minrate[countarrival] = min_rate1[I]
            # Priority set to 1 for the first flow
            userpriority1[countarrival] = userpriority_new
            # Flow number for Adapted Dijsktra set to 1 for first flow
            flownumber_new = flownumber_new + 1
            # Flow number for Multicommodity set to 1 for the first flow


            flow_duration = np.random.exponential(servicetime[I])
            if I <= 3:
                connection_type = 0  # Voice Call
            elif I <= 7:
                connection_type = 1  # Video Call
            elif I <= 11:
                connection_type = 2  # Data Call
            elif I <= 3 + 12:
                connection_type = 0  # Voice Call - Second Cluster
            elif I <= 7 + 12:
                connection_type = 1  # Video Call - Second Cluster
            else:
                connection_type = 2  # Data Call - Second Cluster
            ################################################
            # updateonentry1 does routing using Adapted Dijkstra
            ################################################
            if path_final[0][0] == 0:
                diff = int((flowarrivaltime[I] - time_service) / ( packet_size/ 80000))
                for node_no in range(1, noOfNodes + 1, 1):
                    for next_nodeno in range(0, len(node_links[node_no]), 1):
                        nodes_slot_unused[node_no-1][node_links[node_no][next_nodeno]-1] += diff
                        nodes_slot_total[node_no-1][node_links[node_no][next_nodeno]-1] += diff
                time_service = flowarrivaltime[I]  # If time_service is much lesser than the next flow arrival time , we can fast forward the time as the queue would be empty till flowarrivaltime[I]

            s, d, flow_type, min_rate, \
            flownumber, userpriority, blockstate, \
            blockstate_new, wt_matx, wt_matx_real, \
            wt_matx_real1, path_final = findRoute(p, s, d, flow_type, min_rate, flownumber, userpriority, source[I],
                                                  destination[I], flow_type1[I], min_rate1[I], flownumber_new, userpriority_new,
                                                  path_final, wt_matx, wt_matx_real, wt_matx_real1, blockstate, flow_duration,
                                                  flowarrivaltime[I], connection_type, packet_size, packet_datarate[I],
                                                  header_size)

            usage_matx = np.subtract(np.divide(1,orig_matx), np.divide(1,wt_matx))
            rho_matx = np.divide(usage_matx,np.divide(1,orig_matx))

            rho_matx_sum = rho_matx_sum + np.multiply(new_c - old_c, np.nan_to_num(rho_matx))
            sum_c = sum_c + (new_c - old_c)
            print (np.nanmin(rho_matx))
            print (np.nanmax(rho_matx))
            # Debugging
            # np.savetxt("updweightmatx.csv", 1/wt_matx, delimiter=",")
            # np.savetxt("updorweightmatxreal.csv", 1/wt_matx_real, delimiter=",")
            # np.savetxt("updorweightmatxreal1.csv", 1/wt_matx_real1, delimiter=",")
            # print("INSERT")

            total_matx = np.sum(1 / wt_matx)
            total_real = np.sum(1 / wt_matx_real)
            total_real1 = np.sum(1 / wt_matx_real1)
            k = 0
            weight_diff = 0
            realweight_diff = 0
            while path_final[k][0] != 0:
                j = 15
                while path_final[k][j] != 0 and j != 26:
                    weight_diff = weight_diff + path_final[k][4]
                    if path_final[k][1] == 0:
                        realweight_diff = realweight_diff + path_final[k][4]
                    j += 1
                weight_diff = weight_diff - path_final[k][4]
                if path_final[k][1] == 0:
                    realweight_diff = realweight_diff - path_final[k][4]
                k += 1
            if (orig_total_matx - total_matx - weight_diff) > 2 or (orig_total_matx - total_matx - weight_diff) < -2:
                print "Oops"
            if (orig_total_real1 - total_real1 - realweight_diff) > 2:
                print "Oops"

            if blockstate_new == 0:  # If call is blocked by apadted Dijkstra
                count_algo1 = count_algo1 + 1  # Increase count_algo1 counter and below counters for voice/video/data if tracking statistics started
                if countarrival > start:  # If tracking statistics started
                    # if I <= 3:
                    #     blockedvoice_alog1 = blockedvoice_alog1 + 1
                    # elif I <= 7:
                    #     blockedvideo_algo1 = blockedvideo_algo1 + 1
                    # else:
                    #     blocekednonrealtime_algo1 = blocekednonrealtime_algo1 + 1

                    if I <= 3:
                        blockedvoice_alog1 = blockedvoice_alog1 + 1  # Voice Call
                    elif I <= 7:
                        blockedvideo_algo1 = blockedvideo_algo1 + 1  # Video Call
                    elif I <= 11:
                        blocekednonrealtime_algo1 = blocekednonrealtime_algo1 + 1  # Data Call
                    elif I <= 3 + 12:
                        blockedvoice_alog1 = blockedvoice_alog1 + 1  # Voice Call - Second Cluster
                    elif I <= 7 + 12:
                        blockedvideo_algo1 = blockedvideo_algo1 + 1  # Video Call - Second Cluster
                    else:
                        blocekednonrealtime_algo1 = blocekednonrealtime_algo1 + 1  # Data Call - Second Cluster

            blockstate1[countarrival] = blockstate_new  # blockstate1 counter is updated

            # ########################################## End of Adapted Dijkstra ##########################
            '''
            # ########################################## Routing using multicommodity #####################
            if I <= noSD - 1:
                if I == 0:
                    chosenprob1 = probabilities[0:paths[I]]
                    chosenprob2 = probabilities[3 * total:3 * total + paths[I]]
                else:
                    chosenprob1 = probabilities[paths[I - 1]:paths[I]]
                    chosenprob2 = probabilities[3 * total + paths[I - 1]:3 * total + paths[I]]
            elif I <= 2 * noSD - 1:
                if I == 4:
                    chosenprob1 = probabilities[total:total + paths[I - noSD]]
                    chosenprob2 = probabilities[4 * total:4 * total + paths[I - noSD]]
                else:
                    chosenprob1 = probabilities[total + paths[I - noSD - 1]:total + paths[I - noSD]]
                    chosenprob2 = probabilities[4 * total + paths[I - noSD - 1]:4 * total + paths[I - noSD]]
            else:
                if I == 8:
                    chosenprob1 = probabilities[2 * total:2 * total + paths[I - 2 * noSD]]
                else:
                    chosenprob1 = probabilities[2 * total + paths[I - 2 * noSD - 1]:2 * total + paths[I - 2 * noSD]]
            # ######################################## Routing begins here #################################
            if flow_type1[I] == 0:  # For realtime flow routing ( Voice and Video)
                allo = allocaterealupdated1(p, s_multi, d_multi, flow_type_multi, min_rate_multi, flownumber_multi,userpriority_multi, source[I], destination[I], flow_type1[I], min_rate1[I], flownumber_new_multi,userpriority_new_multi, path_final_multi, wt_matx_multi, wt_matx_real1_multi, wt_matx_real_multi, blockstate_multi, a11, b11, c11, d11, g11, h11, i11, j11, I, chosenprob1, chosenprob2)
                allo.service()
                s_multi = allo.s_multi
                d_multi = allo.d_multi
                flow_type_multi = allo.flow_type_multi
                min_rate_multi = allo.min_rate_multi
                flownumber_multi = allo.flownumber_multi
                userpriority_multi = allo.userpriority_multi
                wt_matx_multi = allo.wt_matx
                wt_matx_real1_multi = allo.wt_matx_real1
                wt_matx_real_multi = allo.wt_matx_real
                blockstate_new_multi = allo.callblock
                path_final_multi = allo.path_final
                blockstate_multi = allo.blockstate_multi

            else:
                aln = allocatenonrealupdated1(p, s_multi, d_multi, flow_type_multi, min_rate_multi, flownumber_multi, userpriority_multi,
                                              source[I], destination[I], flow_type1[I], min_rate1[I], flownumber_new_multi,
                                              userpriority_new_multi, path_final_multi, wt_matx_multi,
                                              blockstate_multi, a11, b11, c11, d11, I, chosenprob1)
                aln.service()
                s_multi = aln.s_multi
                d_multi = aln.d_multi
                flow_type_multi = aln.flow_type_multi
                min_rate_multi = aln.min_rate_multi
                flownumber_multi = aln.flownumber_multi
                userpriority_multi = aln.userpriority_multi
                wt_matx_multi = aln.wt_matx
                blockstate_new_multi = aln.callblock
                path_final_multi = aln.path_final
                blockstate_multi = aln.blockstate_multi

            if blockstate_new_multi == 0:  # If call is blocked by apadted Dijkstra
                count_multi = count_multi + 1  # Increase count_algo1 counter and below counters for voice/video/data if tracking statistics started
                if countarrival > start:  # If tracking statistics started
                    if I <= arrivalratesize / connectiontypes:
                        blockedvoice_multi = blockedvoice_multi + 1
                    elif I <= 2 * arrivalratesize / connectiontypes:
                        blockedvideo_multi = blockedvideo_multi + 1
                    else:
                        blocekednonrealtime_multi = blocekednonrealtime_multi + 1

            blockstate1_multi[countarrival] = blockstate_new_multi  # blockstate1 counter is updated
            '''
            ##############################################
            if countarrival > start:  # Tracking starts here
                # Total counts of various call types arrived so flowarrivaltime
                # if I <= 3:
                #     totalvoice = totalvoice + 1
                # elif I <= 7:
                #     totalvideo = totalvideo + 1
                # else:
                #     totalnonrealtime = totalnonrealtime + 1
                ##############################################

                if I <= 3:
                    totalvoice = totalvoice + 1  # Voice Call
                elif I <= 7:
                    totalvideo = totalvideo + 1 # Video Call
                elif I <= 11:
                    totalnonrealtime = totalnonrealtime + 1  # Data Call
                elif I <= 3 + 12:
                    totalvoice = totalvoice + 1  # Voice Call - Second Cluster
                elif I <= 7 + 12:
                    totalvideo = totalvideo + 1  # Video Call - Second Cluster
                else:
                    totalnonrealtime = totalnonrealtime + 1 # Data Call - Second Cluster

                if blockstate_new == 0:
                    blockalgo1 = blockalgo1 + 1  # Counting number of calls blocked by adapted Dijkstra

            count1departure[countarrival] = countdeparture  # Entries to the number of departures happened so far
            countarrival = countarrival + 1  # Increase Call Counter
            old_c = c
            flowarrivaltime[I] = flowarrivaltime[I] + np.random.exponential(np.divide(1, arrivalrate[I]))
            new_c = flowarrivaltime.min()


# print avgcost1
totalrealtime = totalvoice + totalvideo
fracvoice_algo1 = float(blockedvoice_alog1*1.0/totalvoice)
fracvideo_algo1 = float(blockedvideo_algo1*1.0/totalvideo)
fracnonreal_algo1 = float(blocekednonrealtime_algo1*1.0/totalnonrealtime)
sj_time = sum_soujorn/number_soujorn*1.0
video_endtoend = (Video_e2e/(scale))/Video_e2e_Count
voice_endtoend = (Voice_e2e / (scale)) / Voice_e2e_Count
file_endtoend = (File_e2e/(scale)) / File_e2e_Count
if hetero:
    file_endtoend_onlySatellite = (File_Mean_Total_Time_Satellite / (scale)) / File_Mean_Total_Time_Count_Satellite
    file_endtoend_withoutSatellite = (File_Mean_Total_Time_WO_Satellite / (scale)) / File_Mean_Total_Time_Count_WO_Satellite
    file_endtoend_combined = (File_Mean_Total_Time_All / (scale)) / File_Mean_Total_Time_Count_All
print "Soujorn Time", sj_time
print fracvoice_algo1, fracvideo_algo1, fracnonreal_algo1
print "lambda", lamb
print voice_endtoend, video_endtoend, file_endtoend


if hetero:
    np.savetxt("resultsHetero" + str(lamb) + ".csv", np.array([[lamb, fracvoice_algo1,
                                                          fracvideo_algo1, fracnonreal_algo1, sj_time,
                                                          voice_endtoend,video_endtoend,file_endtoend, file_endtoend_onlySatellite, file_endtoend_withoutSatellite, file_endtoend_combined
                                                          ]]), delimiter=",", fmt="%.10f")
    print file_endtoend_onlySatellite, file_endtoend_withoutSatellite, file_endtoend_combined
else:
    np.savetxt("resultsHetero" + str(lamb) + ".csv", np.array([[lamb, fracvoice_algo1,
                                                                fracvideo_algo1, fracnonreal_algo1, sj_time,
                                                                voice_endtoend, video_endtoend, file_endtoend
                                                                ]]), delimiter=",", fmt="%.10f")