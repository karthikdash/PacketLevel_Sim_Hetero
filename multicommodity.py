import numpy as np
from updateonentry1 import updateonentry1
from updateonexit import updateonexit
from Packets import Packets
import bisect
from allocaterealupdated1 import allocaterealupdated1
from allocatenonrealupdated1 import allocatenonrealupdated1

# Adapted Disjktra Packet Simulator for One Way Video Streaming
lamb = 0.009
limit = 20000
hetero = False
# from udpateonentry import updateonentry
# Network Parameters
connection_type = 0
min_arrivaltime = 0
# Total Number of Nodes
n = 20
p = n
noOfNodes = n
node1packetcounter = 0
nodeoutsidecounter = 0
packetcounter = 0
time_service = 0.1
videofinish = 0
# Link is (link_src[i],link_dest[i])
link_src = [1, 1, 2, 2, 3, 4, 4, 5, 5, 5, 6, 7, 8, 9]
link_dest = [2, 8, 3, 9, 4, 5, 10, 6, 7, 10, 7, 8, 10, 10]

link_src_secondCluster = [1 + 10, 1 + 10, 2 + 10, 2 + 10, 3 + 10, 4 + 10, 4 + 10, 5 + 10, 5 + 10, 5 + 10, 6 + 10, 7 + 10, 8 + 10, 9 + 10, 8]
link_dest_secondCluster = [2 + 10, 8 + 10, 3 + 10, 9 + 10, 4 + 10, 5 + 10, 10 + 10, 6 + 10, 7 + 10, 10 + 10, 7 + 10, 8 + 10, 10 + 10, 10 + 10, 13]


link_src = link_src + link_src_secondCluster
link_dest = link_dest + link_dest_secondCluster

# C[2, 1] = C[1, 2]
# C[4, 1] = C[1, 4]
# C[6, 1] = C[1, 6]
# C[3, 2] = C[2, 3]
# C[4, 2] = C[2, 4]
# C[4, 3] = 5 * (10 ** 6)
# C[5, 3] = C[3, 5]
# C[5, 4] = C[4, 5]
# C[6, 4] = C[4, 6]
# C[6, 5] = 5 * (10 ** 6)

# link_rate_orig = [3, 1, 2, 3, 3, 3, 2]
link_factor = 1
caps = np.append(np.multiply(2, np.ones(28)), 1)
link_rate_orig = np.multiply(link_factor, caps)

# Interference Sets (Links inside a set can tx/rx simultaneously)

'''
inter_sets = {

    0: [[9, 10], [1, 2], [7, 8], [4, 5]],
    1: [[4, 10], [1, 8], [2, 3], [5, 7]],
    2: [[5, 10], [6, 7], [2, 9], [1, 8], [3, 4]],
    3: [[8, 10], [1, 2], [3, 4], [5, 6], [7, 8]],
    4: [[10, 9], [2, 1], [8, 7], [5, 4]],
    5: [[10, 4], [8, 1], [3, 2], [7, 5]],
    6: [[10, 5], [7, 6], [9, 2], [8, 1], [4, 3]],
    7: [[10, 8], [2, 1], [4, 3], [6, 5], [8, 7]],

}



# Multiplication factor for repeated independent sets
inter_multifactor = [link_inter[0] + link_inter[3],
                     link_inter[1] + link_inter[2],
                     link_inter[1],
                     link_inter[2],
                     link_inter[2]+link_inter[3],
                     link_inter[0],
                     link_inter[1],
                     link_inter[3],
                     link_inter[1],
                     link_inter[2],
                     link_inter[2],
                     link_inter[1]+link_inter[3],
                     link_inter[3],
                     link_inter[0]]

inter_multifactor2 = [link_inter[4+0] + link_inter[4+3],
                     link_inter[4+1] + link_inter[4+2],
                     link_inter[4+1],
                     link_inter[4+2],
                     link_inter[4+2]+link_inter[4+3],
                     link_inter[4+0],
                     link_inter[4+1],
                     link_inter[4+3],
                     link_inter[4+1],
                     link_inter[4+2],
                     link_inter[4+2],
                     link_inter[4+1]+link_inter[4+3],
                     link_inter[4+3],
                     link_inter[4+0]]
'''

header_size = 0.00
voice_packet_size = 500.00  # Bits
video_packet_size = voice_packet_size
file_packet_size = voice_packet_size

link_onprob1 = [0.3] * 28 + [0.3]

link_errorprob1 = [0.05] * 28 + [0]

link_onprob2 = [0.3] * 28 + [0.3]

link_errorprob2 = [0.01] * 28 + [0]

link_onprob3 = [0.2] * 28 + [0.4]

link_errorprob3 = [0.07] * 28 + [0]

# Defined in source-destination pairs with rate requirements
source1 = [2, 4, 3, 1]
destination1 = [6, 8, 7, 5]

source_secondCluster = [2 + 10, 4 + 10, 3 + 10, 1 + 10]
destination_secondCluster = [6 + 10, 8 + 10, 7 + 10, 5 + 10]



s5 = 1
s6 = len(source1)
packet_node_count = 0
# Network Parameters
header_size = 0.00
voice_packet_size = 256.00  # Bits
video_packet_size = 256.00
file_packet_size = 256.00
# Service Time is exponentially distributed with mean T
T = 150
# Arrival Rate

scale = 100.0
# (MB/frameSize * [Link Rates corresponding to link_src[i],link_dest[j]]
link_rate = np.multiply((1000000.0/file_packet_size), link_rate_orig)
# link_rate2 = np.multiply((1000000.0/1), link_rate_orig)

pure_link_rate = link_rate_orig

# Data Rate Requirements
data_require = [22, 80, 22, 11, 400, 400, 400, 400, 300, 400, 300, 300]*2 + [300]
# packet_datarate = [22000.0, 80000.0, 22000.0, 11000.0, 400000.0, 400000.0, 400000.0, 400000.0, 300000.0, 400000.0, 300000.0, 300000.0]
packet_datarate = np.multiply(data_require, 1000)
#
# data_require = [22]*12
# packet_datarate = [22000.0]*12

# 232 = frame size - overheads size
min_rate1 = np.multiply(1000.0/232, data_require)
min_rate2 = np.multiply(T*lamb*(1000.0/232), data_require)
flow_type1 = [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2]*2 + [2]
if hetero:
    arrivalrate = np.multiply(lamb, np.ones((12+12+1)))
else:
    arrivalrate = np.multiply(lamb, np.ones((12)))
servicetime = np.append(np.multiply(150, np.ones((8))), np.multiply(7000000, np.ones((4))))
servicetime = servicetime.tolist()*2 + [7000000]
# Video,Voice and Data
connectiontypes = 3


# Iterations (Higher value can lead to long execution times)
# limit = 100000

# Observation from 'start' iteration ?
start = 50
scale = 100.0
# Effective weight after factoring in on-off states and frame error probability
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
    A[q][r] = weight[i]
    # print A[q][r]
    A[r][q] = weight[i]
    # print A[r][q]
eff_capacity_matx = A
a = np.delete(eff_capacity_matx, 0, 0)
a = np.delete(a, 0, 1)


# Origianl Capacities of Links
B = np.zeros((n, n))
for i in range(0, m1):
    q = link_src[i] - 1
    r = link_dest[i] - 1
    B[q][r] = pure_link_rate[i]
    B[r][q] = pure_link_rate[i]
b = np.delete(B, 0, 0)
b = np.delete(B, 0, 1)

# C gives the retransmission probability for packet level
C = np.zeros((n, n))
for i in range(0, m1):
    q = link_src[i] - 1
    r = link_dest[i] - 1
    C[q][r] = weight_x[i]
    C[r][q] = weight_x[i]
c = np.delete(C, 0, 0)
c = np.delete(c, 0, 1)

# To ignore Division by O warnings. /0 taken as Inf
with np.errstate(divide='ignore', invalid='ignore'):
    # Adapted Dijkstra Weights
    # wt_matx = np.divide(1.0526, eff_capacity_matx)
    wt_matx = np.divide(1, eff_capacity_matx)
    wt_matx_real = np.divide(1, eff_capacity_matx)
    wt_matx_real1 = np.divide(1.33, eff_capacity_matx)
    # Multicommodity Weights
    wt_matx_multi = np.divide(1, eff_capacity_matx)
    wt_matx_real_multi = np.divide(1, eff_capacity_matx)
    wt_matx_real1_multi = np.divide(1.33, eff_capacity_matx)
    # Enhanced Dijkstra Wights
    wt_matx_block = np.divide(1, eff_capacity_matx)
    wt_matx_real_block = np.divide(1, eff_capacity_matx)
    wt_matx_real1_block = np.divide(1.33, eff_capacity_matx)

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
# np.savetxt("orweightmatx.csv", 1/wt_matx, delimiter=",")
# np.savetxt("orweightmatxreal.csv", 1/wt_matx_real, delimiter=",")
# np.savetxt("orweightmatxreal1.csv", 1/wt_matx_real1, delimiter=",")

orig_total_matx = np.sum(1/wt_matx)
orig_total_real = np.sum(1/wt_matx_real)
orig_total_real1 = np.sum(1/wt_matx_real1)

# To get Source-Destination Pairs defined by problem statement

# for i in range(0, connectiontypes):
#     source = np.append(source, source1)
#     destination = np.append(destination, destination1)

# source_secondCluster = [2 + 10, 4 + 10, 3 + 10, 1 + 10, 6 + 10, 5 + 10,]
# destination_secondCluster = [6 + 10, 8 + 10, 7 + 10, 5 + 10, 2 + 10, 1 + 10]

source = source1 + source1 + source1 + source_secondCluster + source_secondCluster + source_secondCluster + [3]
destination = destination1 + destination1 + destination1 + destination_secondCluster + destination_secondCluster + destination_secondCluster + [ 3 + 10]

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

# Multicommodity Variable Defination
s_multi = []
d_multi = []
flow_type_multi = []
min_rate_multi = []
flownumber_multi = []
userpriority_multi = []
blockstate_multi = []
userpriority_new_multi = 1
flownumber_new_multi = 0

# Enhanced Dijkstra Variable Definations
s_block = []
d_block = []
flow_type_block = []
flownumber_block = []
userpriority_block = []
blockstate_block = []
userpriority_new_block = 1
flownumber_new_block = 0

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
path_final = np.zeros((3*limit, p+16))
path_final_multi = np.zeros((3*limit, p+5))
path_final_block = np.zeros((3*limit, p+5))

count1 = np.zeros((limit))
coutn1withmulti = np.zeros((limit))
count1withblock = np.zeros((limit))
count1departure = np.zeros((limit))
frac = np.zeros((limit-start))
frawwithmulti = np.zeros((limit-start))
fracwithblock = np.zeros((limit-start))
blockfirstattempt1 = np.zeros((limit-start))
blockfirstattempt = 0
countvoice = 0
countvideo = 0
countnonrealtime = 0
NBalgo1_Balgo2 = 0
Balgo1_Nbalgo2 = 0
NBalgo2_Bmulti = 0
Balgo2_NBmulti = 0
NBalgo1_Bmulti = 0
Balgo1_NBmulti = 0
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

# Input Parameters for Multicommodity Algorithm #

# Demand fractions are obtained by solving convex optimization through Frank Wolfe Alogorithm
fractions = np.genfromtxt('xfractions1.csv')
a11 = np.genfromtxt('A.csv', delimiter=',')
b11 = np.genfromtxt('B.csv', delimiter=',')
c11 = np.genfromtxt('C.csv', delimiter=',')
d11 = np.genfromtxt('D.csv', delimiter=',')
g11 = np.genfromtxt('G.csv', delimiter=',')
h11 = np.genfromtxt('H.csv', delimiter=',')
i11 = np.genfromtxt('I.csv', delimiter=',')
j11 = np.genfromtxt('J.csv', delimiter=',')
[m1, n1] = np.shape(a11)
[m2, n2] = np.shape(b11)
[m3, n3] = np.shape(c11)
[m4, n4] = np.shape(d11)
paths = [m1, m1+m2, m1+m2+m3, m1+m2+m3+m4]
pathsno = [m1, m2, m3, m4]
total = m1+m2+m3+m4
fractionssize1 = np.shape((fractions))[0]
fractionssize = 1
probabilities = np.zeros((fractionssize1, fractionssize))

# for i in range(0, m1, 1):
#     probabilities[i] = fractions[i]/min_rate2[0]
#     probabilities[total + i] = fractions[total + i]/min_rate2[s6 + 0]
#     probabilities[2*total + i] = fractions[2*total + i]/min_rate2[2*s6 + 0]
#     probabilities[3*total + i] = fractions[3*total + i]/min_rate2[0]
#     probabilities[4*total + i] = fractions[4*total + i]/min_rate2[s6 + 0]
# for i in range(0, m2, 1):
#     probabilities[m1 + i] = fractions[m1 + i]/min_rate2[1]
#     probabilities[total + m1 + i] = fractions[total + m1 + i]/min_rate2[s6 + 1]
#     probabilities[2*total + m1 + i] = fractions[2*total + m1 + i]/min_rate2[2*s6 + 1]
#     probabilities[3*total + m1 + i] = fractions[3*total + m1 + i]/min_rate2[1]
#     probabilities[4*total + m1 + i] = fractions[4*total + m1 + i]/min_rate2[s6 + 1]
# for i in range(0, m3, 1):
#     probabilities[m1 + m2 + i] = fractions[m1 + i]/min_rate2[2]
#     probabilities[total + m1 + m2 + i] = fractions[total + m1 + m2 + i]/min_rate2[s6 + 2]
#     probabilities[2*total + m1 + m2 + i] = fractions[2*total + m1 + m2 + i]/min_rate2[2*s6 + 2]
#     probabilities[3*total + m1 + m2 + i] = fractions[3*total + m1 + m2 + i]/min_rate2[2]
#     probabilities[4*total + m1 + m2 + i] = fractions[4*total + m1 + m2 + i]/min_rate2[s6 + 2]
# for i in range(0, m4, 1):
#     probabilities[m1 + m2 + m3 + i] = fractions[m1 + m2 + m3 + i]/min_rate2[3]
#     probabilities[total + m1 + m2 + m3 + i] = fractions[total + m1 + m2 + m3 + i]/min_rate2[s6 + 3]
#     probabilities[2*total + m1 + m2 + m3 + i] = fractions[2*total + m1 + m2 + m3 + i]/min_rate2[2*s6 + 3]
#     probabilities[3*total + m1 + m2 + m3 + i] = fractions[3*total + m1 + m2 + m3 + i]/min_rate2[3]
#     probabilities[4*total + m1 + m2 + m3 + i] = fractions[4*total + m1 + m2 + m3 + i]/min_rate2[s6 + 3]

# ##
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

'''
flow_tag = 0  -> voice
flow_tag = 1  -> video
flow_tag = 2  -> file / non-realtime data
'''

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
for k1 in range(0, len(link_src) -1, 1):
    nodes_nonreal[link_src[k1], link_dest[k1]] = [[0, 0, 0, 0, 0, 0, 0, 0], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    nodes_nonreal[link_dest[k1], link_src[k1]] = [[0, 0, 0, 0, 0, 0, 0, 0], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    nodes_real[link_src[k1], link_dest[k1]] = []
    nodes_real[link_dest[k1], link_src[k1]] = []
nodes_nonreal[link_src[k1+1], link_dest[k1+1]] = [[0, 0, 0, 0, 0, 0, 0, 0], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
nodes_real[link_src[k1+1], link_dest[k1+1]] = []

# link_src = [1, 1, 2, 2, 3, 4, 4,  5, 5, 5,  6, 7,  8, 9]
# link_dest =[2, 8, 3, 9, 4, 5, 10, 6, 7, 10, 7, 8, 10, 10]
node_links = {
    1: [2, 8],
    2: [3, 9, 1],
    3: [4, 2],
    4: [5, 10, 3],
    5: [6, 7, 10, 4],
    6: [7, 5],
    7: [8, 5, 6],
    8: [10, 1, 7, 13],
    9: [10, 2],
    10: [4, 5, 8, 9],
    1 + 10: [2 + 10, 8 + 10],
    2 + 10: [3 + 10, 9 + 10, 1 + 10],
    3 + 10: [4 + 10, 2 + 10],
    4 + 10: [5 + 10, 10 + 10, 3 + 10],
    5 + 10: [6 + 10, 7 + 10, 10 + 10, 4 + 10],
    6 + 10: [7 + 10, 5 + 10],
    7 + 10: [8 + 10, 5 + 10, 6 + 10],
    8 + 10: [10 + 10, 1 + 10, 7 + 10],
    9 + 10: [10 + 10, 2 + 10],
    10 + 10: [4 + 10, 5 + 10, 8 + 10, 9 + 10],

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

# Voice Mean Delays at each node
Voice_Mean_Time0 = 0
Voice_Mean_Time1 = 0
Voice_Mean_Time2 = 0
Voice_Mean_Time3 = 0
Voice_Mean_Time4 = 0
Voice_Mean_Time5 = 0
Voice_Mean_Time6 = 0
Voice_Mean_Time7 = 0
Voice_Mean_Time8 = 0
Voice_Mean_Time9 = 0
Voice_Mean_Time10 = 0

# Dictionary for easy access of above Means
Voice_Mean_Time = {
    1: Voice_Mean_Time1,
    2: Voice_Mean_Time2,
    3: Voice_Mean_Time3,
    4: Voice_Mean_Time4,
    5: Voice_Mean_Time5,
    6: Voice_Mean_Time6,
    7: Voice_Mean_Time7,
    8: Voice_Mean_Time8,
    9: Voice_Mean_Time9,
    10: Voice_Mean_Time10
}
Video_Mean_Time0 = 0
Video_Mean_Time1 = 0
Video_Mean_Time2 = 0
Video_Mean_Time3 = 0
Video_Mean_Time4 = 0
Video_Mean_Time5 = 0
Video_Mean_Time6 = 0
Video_Mean_Time7 = 0
Video_Mean_Time8 = 0
Video_Mean_Time9 = 0
Video_Mean_Time10 = 0

Video_Mean_Time = {
    1: Video_Mean_Time1,
    2: Video_Mean_Time2,
    3: Video_Mean_Time3,
    4: Video_Mean_Time4,
    5: Video_Mean_Time5,
    6: Video_Mean_Time6,
    7: Video_Mean_Time7,
    8: Video_Mean_Time8,
    9: Video_Mean_Time9,
    10: Video_Mean_Time10,
}

File_Mean_Time0 = 0
File_Mean_Time1 = 0
File_Mean_Time2 = 0
File_Mean_Time3 = 0
File_Mean_Time4 = 0
File_Mean_Time5 = 0
File_Mean_Time6 = 0
File_Mean_Time7 = 0
File_Mean_Time8 = 0
File_Mean_Time9 = 0
File_Mean_Time10 = 0

File_Mean_Time = {
    1: File_Mean_Time1,
    2: File_Mean_Time2,
    3: File_Mean_Time3,
    4: File_Mean_Time3,
    5: File_Mean_Time5,
    6: File_Mean_Time6,
    7: File_Mean_Time7,
    8: File_Mean_Time8,
    9: File_Mean_Time9,
    10: File_Mean_Time10,
}
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

# call_duration = 1.0 / 150

# ################### End Of Packet level Simulation variables############

# Exponential Random distribution at mean 1/lambda
# All the arrival times are computed here. So each flowarrival[] time corresponds to particular source[]
flowarrivaltime = np.random.exponential(np.divide(1, arrivalrate))
arrivalratesize = np.shape(arrivalrate)[0]
arrivalratesize1 = 1
intialArrival = 0

# Function to check if a particular node has packets

def nodesHavePackets():
    for no in range(1, noOfNodes + 1, 1):
        for nextnode in range(0, len(node_links[no]), 1):
            if len(nodes_real[(no, node_links[no][nextnode])]) != 0:
                return True
            for ii in range(0, len(nodes_nonreal[(no, node_links[no][nextnode])])):
                if len(nodes_nonreal[(no, node_links[no][nextnode])][ii]) != 0:
                    return True
    return False

# Function to check if a particular node has realtime packets

def realnodesHavePackets():
    for no in range(1, noOfNodes + 1, 1):
        for nextnode in range(0, len(node_links[no]), 1):
            if len(nodes_real[(no, node_links[no][nextnode])]) != 0:
                return True
    return False

# Function to check if a particular node has nonrealtme packets
def nonrealnodesHavePackets():
    for no in range(1, noOfNodes + 1, 1):
        for nextnode in range(0, len(node_links[no]), 1):
            for ii in range(0, len(nodes_nonreal[(no, node_links[no][nextnode])])):
                if len(nodes_nonreal[(no, node_links[no][nextnode])][ii]) != 0:
                    return True
    return False

def dataPacketsAvailable(source, dest):
    for ii in range(1, len(nodes_nonreal[(source, dest)])):
        if len(nodes_nonreal[(source, dest)][ii]) != 0:
            return True
    return False

def currentNonrealQueue(source, dest):
    no_of_active = 0
    active_indices = []
    inde = 0
    for ii in range(1, len(nodes_nonreal[(source, dest)])):
        if len(nodes_nonreal[(source, dest)][ii]) != 0:
            no_of_active += 1
            active_indices.append(ii)
    return np.random.choice(active_indices)
c = 0


def appendNonRealQueue(source, dest, inde):
    flag = 1
    for jj in range(1, len(nodes_nonreal[(nodes_nonreal[(source, dest)][current_nr_index][0].d_new, nodes_nonreal[(source, dest)][inde][0].path[1])])):
        if len(nodes_nonreal[(nodes_nonreal[(source, dest)][current_nr_index][0].d_new, nodes_nonreal[(source, dest)][inde][0].path[1])][jj]) != 0:
            if nodes_nonreal[(nodes_nonreal[(source, dest)][current_nr_index][0].d_new, nodes_nonreal[(source, dest)][inde][0].path[1])][jj][0].flownumber == \
                    nodes_nonreal[(source, dest)][current_nr_index][0].flownumber:
                if nodes_nonreal[(source, dest)][current_nr_index][0].d_new == 8 and nodes_nonreal[(source, dest)][inde][0].path[1] == 13:
                    init_arrival_time_add = 25  # adding 250 ms (250 ms * 100 because of scale factor) propogation delay for satellite
                else:
                    init_arrival_time_add = 0
                bisect.insort_left(nodes_nonreal[(nodes_nonreal[(source, dest)][current_nr_index][0].d_new, nodes_nonreal[(source, dest)][inde][0].path[1])][jj],
                                   Packets(nodes_nonreal[(source, dest)][inde][0].initial_arrival_time,
                                           nodes_nonreal[(source, dest)][inde][0].service_end_time + init_arrival_time_add,
                                           nodes_nonreal[(source, dest)][inde][0].flow_duration,
                                           nodes_nonreal[(source, dest)][inde][0].flow_tag,
                                           nodes_nonreal[(source, dest)][inde][0].path,
                                           nodes_nonreal[(source, dest)][inde][0].flownumber,
                                           nodes_nonreal[(source, dest)][inde][0].noofpackets,
                                           nodes_nonreal[(source, dest)][inde][0].direction,
                                           nodes_nonreal[(source, dest)][inde][0].node_service_rate,
                                           nodes_nonreal[(source, dest)][inde][0].total_slot_time,
                                           nodes_nonreal[(source, dest)][inde][0].total_slots))
                return
            # else:
            #     if len(nodes_nonreal[(nodes_nonreal[(source, dest)][current_nr_index][0].d_new, nodes_nonreal[(source, dest)][inde][0].path[1])][jj + 1]) == 0:
            #         bisect.insort_left(nodes_nonreal[(nodes_nonreal[(source, dest)][current_nr_index][0].d_new, nodes_nonreal[(source, dest)][inde][0].path[1])][jj+1],
            #                            Packets(nodes_nonreal[(source, dest)][inde][0].initial_arrival_time,
            #                                    nodes_nonreal[(source, dest)][inde][0].service_end_time,
            #                                    nodes_nonreal[(source, dest)][inde][0].flow_duration,
            #                                    nodes_nonreal[(source, dest)][inde][0].flow_tag,
            #                                    nodes_nonreal[(source, dest)][inde][0].path,
            #                                    nodes_nonreal[(source, dest)][inde][0].flownumber,
            #                                    nodes_nonreal[(source, dest)][inde][0].noofpackets,
            #                                    nodes_nonreal[(source, dest)][inde][0].direction,
            #                                    nodes_nonreal[(source, dest)][inde][0].node_service_rate,
            #                                    nodes_nonreal[(source, dest)][inde][0].total_slot_time,
            #                                    nodes_nonreal[(source, dest)][inde][0].total_slots))
            #        break
            flag = 0
    if flag == 1:
        for jj in range(1, len(nodes_nonreal[(nodes_nonreal[(source, dest)][current_nr_index][0].d_new, nodes_nonreal[(source, dest)][inde][0].path[1])])):
            if len(nodes_nonreal[(nodes_nonreal[(source, dest)][current_nr_index][0].d_new, nodes_nonreal[(source, dest)][inde][0].path[1])][jj]) == 0:
                if nodes_nonreal[(source, dest)][current_nr_index][0].d_new == 8 and nodes_nonreal[(source, dest)][inde][0].path[1] == 13:
                    init_arrival_time_add = 25  # adding 250 ms (250 ms * 100 because of scale factor) propogation delay for satellite
                else:
                    init_arrival_time_add = 0
                bisect.insort_left(nodes_nonreal[(nodes_nonreal[(source, dest)][current_nr_index][0].d_new, nodes_nonreal[(source, dest)][inde][0].path[1])][jj],
                                   Packets(nodes_nonreal[(source, dest)][inde][0].initial_arrival_time,
                                           nodes_nonreal[(source, dest)][inde][0].service_end_time + init_arrival_time_add,
                                           nodes_nonreal[(source, dest)][inde][0].flow_duration,
                                           nodes_nonreal[(source, dest)][inde][0].flow_tag,
                                           nodes_nonreal[(source, dest)][inde][0].path,
                                           nodes_nonreal[(source, dest)][inde][0].flownumber,
                                           nodes_nonreal[(source, dest)][inde][0].noofpackets,
                                           nodes_nonreal[(source, dest)][inde][0].direction,
                                           nodes_nonreal[(source, dest)][inde][0].node_service_rate,
                                           nodes_nonreal[(source, dest)][inde][0].total_slot_time,
                                           nodes_nonreal[(source, dest)][inde][0].total_slots))
                return
    # print "nononon"
    for jj in range(1, len(nodes_nonreal[(nodes_nonreal[(source, dest)][current_nr_index][0].d_new, nodes_nonreal[(source, dest)][inde][0].path[1])])):
        if len(nodes_nonreal[(nodes_nonreal[(source, dest)][current_nr_index][0].d_new, nodes_nonreal[(source, dest)][inde][0].path[1])][jj]) == 0:
            if nodes_nonreal[(source, dest)][current_nr_index][0].d_new == 8 and nodes_nonreal[(source, dest)][inde][0].path[1] == 13:
                init_arrival_time_add = 25  # adding 250 ms (250 ms * 100 because of scale factor) propogation delay for satellite
            else:
                init_arrival_time_add = 0
            bisect.insort_left(nodes_nonreal[(nodes_nonreal[(source, dest)][current_nr_index][0].d_new, nodes_nonreal[(source, dest)][inde][0].path[1])][jj],
                               Packets(nodes_nonreal[(source, dest)][inde][0].initial_arrival_time,
                                       nodes_nonreal[(source, dest)][inde][0].service_end_time + init_arrival_time_add,
                                       nodes_nonreal[(source, dest)][inde][0].flow_duration,
                                       nodes_nonreal[(source, dest)][inde][0].flow_tag,
                                       nodes_nonreal[(source, dest)][inde][0].path,
                                       nodes_nonreal[(source, dest)][inde][0].flownumber,
                                       nodes_nonreal[(source, dest)][inde][0].noofpackets,
                                       nodes_nonreal[(source, dest)][inde][0].direction,
                                       nodes_nonreal[(source, dest)][inde][0].node_service_rate,
                                       nodes_nonreal[(source, dest)][inde][0].total_slot_time,
                                       nodes_nonreal[(source, dest)][inde][0].total_slots))
            return



while(countarrival < limit - 1):
    print countarrival, "countarrival", packetcounter , time_service
    if countarrival > 200:
        # fracvideo_algo1 = float(blockedvideo_algo1 * 1.0 / totalvideo)
        # fracnonreal_algo1 = float(blocekednonrealtime_algo1 * 1.0 / totalnonrealtime)
        fracvoice_algo1 = float(blockedvoice_alog1 * 1.0 / totalvoice)
        fracvideo_algo1 = float(blockedvideo_algo1 * 1.0 / totalvideo)
        fracnonreal_algo1 = float(blocekednonrealtime_algo1 * 1.0 / totalnonrealtime)
        sj_time = sum_soujorn / number_soujorn * 1.0
        video_endtoend = (Video_e2e / (scale)) / Video_e2e_Count
        voice_endtoend = (Voice_e2e / (scale)) / Voice_e2e_Count
        file_endtoend = (File_e2e / (scale)) / File_e2e_Count
        print lamb, fracvoice_algo1, fracvideo_algo1, fracnonreal_algo1, voice_endtoend, video_endtoend, file_endtoend, sj_time
        if hetero:
            print (File_Mean_Total_Time_Satellite/ (scale)) / File_Mean_Total_Time_Count_Satellite, \
                (File_Mean_Total_Time_WO_Satellite / (scale)) / File_Mean_Total_Time_Count_WO_Satellite, \
                (File_Mean_Total_Time_All / (scale)) / File_Mean_Total_Time_Count_All


    # We find the minimum get the first arriving flow and hence source node for that corresponding time
    c = flowarrivaltime.min()  # Minimum Value
    I = flowarrivaltime.argmin()  # Index of the Minimum Value
    if countarrival == 0:

        timepreviuos1 = flowarrivaltime[I]  # First Flow arrival time

        # Arrivaltime vector is updated by appending the current flow arrival time which just arrived
        arrivaltime = np.append(arrivaltime, flowarrivaltime[I])

        # <PDQ> Time of departure udpated for the flow that as arrived
        # timeofdeparture = c + np.random.exponential(servicetime[I])

        # DepartureTime vector is updated by appending the current flow departure time which just arrived
        # departuretime = np.append(departuretime, timeofdeparture)
        # departuretime1 = np.append(departuretime1, timeofdeparture)

        # New Arrival time computation for the next flow
        # IMP NOTE: The next flowarrivaltime[I] is calculated at the end of the
        # block

        intialArrival = flowarrivaltime[I]

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

        # Flow number for Multicommodity set to 1 for the first flow
        flownumber_new_multi = flownumber_new_multi + 1

        # Flow number for Enhanced Adapted Dijkstra set to 1 for the first flow
        flownumber_new_block = flownumber_new_block + 1
        # flow_duration = np.random.exponential(servicetime[I])
        flow_duration = np.random.exponential(servicetime[I])

        ########## ADAPTED DIJSKTRA ROUTING ALGORITHM #########################


        # updateonentry1 does routing using Adapted Dijkstra
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

        updateonentry2 = updateonentry1(p, s, d, flow_type, min_rate, flownumber, userpriority, source[I],
                                        destination[I], flow_type1[I], min_rate1[I], flownumber_new, userpriority_new,
                                        path_final, wt_matx, wt_matx_real, wt_matx_real1, blockstate, flow_duration,
                                        flowarrivaltime[I], connection_type, voice_packet_size, packet_datarate[I],
                                        header_size)
        updateonentry2.execute()
        s = updateonentry2.s
        d = updateonentry2.d
        flow_type = updateonentry2.flow_type
        min_rate = updateonentry2.min_rate
        flownumber = updateonentry2.flownumber
        userpriority = updateonentry2.userpriority
        blockstate_new = updateonentry2.blockstate_new
        wt_matx = updateonentry2.wt_matx
        wt_matx_real = updateonentry2.wt_matx_real
        wt_matx_real1 = updateonentry2.wt_matx_real1
        path_final = updateonentry2.path_final
        blockstate = updateonentry2.blockstate

        # For rho calculations
        old_c = 0
        c = flowarrivaltime.min()
        new_c = c

        # Debugging

        # np.savetxt("updweightmatx.csv", 1/ wt_matx, delimiter=",")
        # np.savetxt("updorweightmatxreal.csv", 1/ wt_matx_real, delimiter=",")
        # np.savetxt("updorweightmatxreal1.csv", 1/ wt_matx_real1, delimiter=",")

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
        # if (orig_total_matx - total_matx - weight_diff) > 2 or (orig_total_matx - total_matx - weight_diff) < -2 :
        #     print "Oops"
        # if (orig_total_real1 - total_real1 - realweight_diff) > 2:
        #     print "Oops"


        if blockstate_new == 0:  # If call is blocked by apadted Dijkstra
            count_algo1 = count_algo1 + 1
        blockstate1[countarrival] = blockstate_new

        '''
        if blockstate[-1] == 1:  # If last call not blocked

            if I < arrivalratesize/connectiontypes:
                path_final[int(flownumber[-1]) - 1, 3] = 0
                if int(packet_datarate[I] / 100 / voice_packet_size) < 1:
                    path_final[int(flownumber[-1])-1, 2] = int(flow_duration) * 1
                    path_final[int(flownumber[-1]), 2] = int(flow_duration) * 1
                else:
                    path_final[int(flownumber[-1]) - 1, 2] = int(flow_duration) * int(packet_datarate[I] / 100 / voice_packet_size)
                    path_final[int(flownumber[-1]), 2] = int(flow_duration) * int(packet_datarate[I] / 100 / voice_packet_size)
                path_final[int(flownumber[-1]) - 1, 8] = packet_datarate[I]/100

                # Backward Path Packetisation
                path_final[int(flownumber[-1]), 3] = 0

                path_final[int(flownumber[-1]), 8] = packet_datarate[I] / 100
            elif I < 2*arrivalratesize/connectiontypes:
                # Back Path Packetisation
                path_final[int(flownumber[-1]) - 1, 3] = 1
                path_final[int(flownumber[-1]) - 1, 2] = int(flow_duration) * int(packet_datarate[I]/100/video_packet_size)
                path_final[int(flownumber[-1]) - 1, 8] = packet_datarate[I] / 100
                path_final[int(flownumber[-1]), 3] = 1
                path_final[int(flownumber[-1]), 2] = int(flow_duration) * int(packet_datarate[I]/100/video_packet_size)
                path_final[int(flownumber[-1]), 8] = packet_datarate[I] / 100
            else:
                if int(flow_duration*150*1000 / file_packet_size) < 1:
                    file_limit = 1
                else:
                    file_limit = int(flow_duration*150*1000 / file_packet_size)
                path_final[int(flownumber[-1]) - 1, 2] = file_limit
                path_final[int(flownumber[-1]) - 1, 8] = packet_datarate[I] / 100
                path_final[int(flownumber[-1]) - 1, 3] = 2
        '''
        # print blockstate, "blockstate"
        # print path_final, "pathfinalstate"
        # print flow_type1[I]

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
        # finished their service (nodesHavePackets() == false)

        # c1 = departuretime1.min()  # Minimum Value
        # I1 = departuretime1.argmin()  # Index of the Minimum Value
        if min_arrivaltime < c and path_final[0][0] != 0:  # and nodesHavePackets():
            # Packetization
            kk = 0
            while min_arrivaltime < c and path_final[0][0] != 0:  # and nodesHavePackets():
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
                    # if min_arrivaltime <= time_service + file_packet_size / 20000:
                    if True:
                        #if path_final[k][3] == 0 and (path_final[k][6] - float(1.0 / ((path_final[k][8]) / voice_packet_size))) <= (time_service + file_packet_size / 20000):
                        if path_final[k][3] == 0 and (path_final[k][6] <= time_service + file_packet_size / 20000):
                            nodes_real[(int(path_final[k][15]), int(path_final[k][16]))].append(Packets(
                                path_final[k][6],
                                path_final[k][6],
                                path_final[k][7], 0, path_final[k][15:25].tolist() + [0], path_final[k][0],
                                path_final[k][2], True, path_final[k][8], 0, 0))
                            if countarrival == 1:
                                time_service = path_final[k][6]
                            path_final[k][6] = path_final[k][6] + float(1.0 / ((path_final[k][8]) / voice_packet_size))

                            k += 1

                            nodes_real[(int(path_final[k][15]), int(path_final[k][16]))].append(Packets(
                                path_final[k][6],
                                path_final[k][6],
                                path_final[k][7], 0, path_final[k][15:25].tolist() + [0], path_final[k][0],
                                path_final[k][2], False, path_final[k][8], 0, 0))
                            path_final[k][6] = path_final[k][6] + float(1.0 / ((path_final[k][8]) / voice_packet_size))
                            k += 1
                        #elif path_final[k][3] == 1 and (path_final[k][6] - float(1.0 / ((path_final[k][8]) / voice_packet_size))) <= (time_service + file_packet_size / 20000):  # Video Calls
                        elif path_final[k][3] == 1 and (path_final[k][6] <= time_service + file_packet_size / 20000):
                            nodes_real[(int(path_final[k][15]), int(path_final[k][16]))].append(Packets(
                                path_final[k][6],
                                path_final[k][6],
                                path_final[k][7], 1, path_final[k][15:25].tolist() + [0], path_final[k][0],
                                path_final[k][2], True, path_final[k][8], 0, 0))
                            # nodes_total_real_packets[int(path_final[k][11])-1][int(path_final[k][12])-1] += 1
                            if countarrival == 1:
                                time_service = path_final[k][6]
                            path_final[k][6] = path_final[k][6] + float(1.0 / ((path_final[k][8]) / video_packet_size))
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
                            #     1.0 / ((path_final[k][8]) / video_packet_size))
                            k += 1
                            if countarrival == 1:
                                time_service = path_final[0][6]
                                # min_arrivaltime = time_service
                        else:
                            k += 1
                    elif nodesHavePackets() == False:
                        time_service = time_service + file_packet_size / 20000
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
                    if nodesHavePackets() == False:
                        time_service = time_service + file_packet_size / 80000
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
                                        nodeoutsidecounter += 1
                                        nodes_slot_used[node_no-1][node_links[node_no][next_nodeno]-1] += 1
                                        nodes_slot_total[node_no-1][node_links[node_no][next_nodeno]-1] += 1
                                        # if nodes_real[(node_no, node_links[node_no][next_nodeno])][0].arrival_time >= max(serviceend_time):
                                        '''
                                        arrivaltimes = []
                                        for nd_no in range(1, noOfNodes + 1, 1):
                                            if len(nodes_real[str(nd_no)]) == 0:
                                                arrivaltimes.append(0)
                                            else:
                                                arrivaltimes.append(nodes_real[str(nd_no)][0].arrival_time)
                                        '''
                                        s_link = int(nodes_real[(node_no, node_links[node_no][next_nodeno])][0].path[0])
                                        d_link = int(nodes_real[(node_no, node_links[node_no][next_nodeno])][0].path[1])
                                        #if nodes_real[(node_no, node_links[node_no][next_nodeno])][0].arrival_time <= time_service and B[s_link - 1][d_link - 1] == 8:
                                        if B[s_link - 1][d_link - 1] == 8:
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
                                                nodes_real[(node_no, node_links[node_no][next_nodeno])][packetno].addSlotDelay(file_packet_size / 80000)

                                            nodes_slot_queue_real_len[node_no-1][node_links[node_no][next_nodeno]-1] += len(nodes_real[(node_no, node_links[node_no][next_nodeno])])
                                            nodes_real[(node_no, node_links[node_no][next_nodeno])][0].service(
                                                max(nodes_real[(node_no, node_links[node_no][next_nodeno])][0].arrival_time, time_service),
                                                B[s_link - 1][d_link - 1], False, link_retransmit_prob,
                                                file_packet_size / 80000)
                                            # Appending to the serving Queue
                                            # serviceend_time[node_no] = nodes_real[(node_no, node_links[node_no][next_nodeno])][0].service_end_time
                                            # packets_tracker[str(node_no)].append(nodes_real[(node_no, node_links[node_no][next_nodeno])][0])
                                            if link_retransmit_prob == 1:
                                                if nodes_real[(node_no, node_links[node_no][next_nodeno])][0].flow_tag == 0:
                                                    Voice_Mean_Time[node_no] = (Voice_Mean_Time[node_no] +
                                                                                nodes_real[(node_no, node_links[node_no][next_nodeno])][
                                                                                    0].service_end_time -
                                                                                nodes_real[(node_no, node_links[node_no][next_nodeno])][0].arrival_time) / 2.0
                                                    # if Voice_Mean_Time[node_no] > 1:
                                                        # print "no"
                                                else:
                                                    Video_Mean_Time[node_no] = (Video_Mean_Time[node_no] +
                                                                                nodes_real[(node_no, node_links[node_no][next_nodeno])][
                                                                                    0].service_end_time -
                                                                                nodes_real[(node_no, node_links[node_no][next_nodeno])][0].arrival_time) / 2.0
                                                    #if Video_Mean_Time[node_no] > 1:
                                                        # print "no"
                                                # Appending to the next node receiving Queue
                                                if nodes_real[(node_no, node_links[node_no][next_nodeno])][0].d_new == 99:
                                                    if nodes_real[(node_no, node_links[node_no][next_nodeno])][0].flow_tag == 0:
                                                        ################### Decrementing the packet count from path_final ############
                                                        k = 0
                                                        if nodes_real[(node_no, node_links[node_no][next_nodeno])][0].direction == True:
                                                            # path_final[nodes_real[(node_no, node_links[node_no][next_nodeno])][0].flownumber ][2] -= 1
                                                            while path_final[k][0] != 0:
                                                                if path_final[k][0] == nodes_real[(node_no, node_links[node_no][next_nodeno])][0].flownumber:
                                                                    path_final[k][2] -= 1
                                                                    path_final[k][9] = (path_final[k][9] + nodes_real[(node_no, node_links[node_no][next_nodeno])][0].total_slot_time) / 2.0
                                                                    Voice_e2e += time_service - nodes_real[(node_no, node_links[node_no][next_nodeno])][0].initial_arrival_time
                                                                    Voice_e2e_Count += 1
                                                                    packetcounter += 1
                                                                    if path_final[k][2] < 1:
                                                                        # print "finished"
                                                                        Voice_Mean_Total = (Voice_Mean_Total + ((path_final[k][9]+path_final[k+1][9]) / 2.0)) / 2.0
                                                                        if True:
                                                                            upde = updateonexit(p, s, d, flow_type, min_rate,
                                                                                                flownumber, userpriority,
                                                                                                path_final[k][0], path_final,
                                                                                                wt_matx,
                                                                                                wt_matx_real,
                                                                                                wt_matx_real1, blockstate)
                                                                            upde.execute()
                                                                            s = upde.s
                                                                            d = upde.d
                                                                            flow_type = upde.flow_type
                                                                            min_rate = upde.min_rate
                                                                            flownumber = upde.flownumber
                                                                            userpriority = upde.userpriority
                                                                            wt_matx = upde.wt_matx
                                                                            wt_matx_real = upde.wt_matx_real
                                                                            wt_matx_real1 = upde.wt_matx_real1
                                                                            path_final = upde.path_final
                                                                            blockstate = upde.blockstate

                                                                            # Debugging
                                                                            # np.savetxt("updweightmatx.csv", 1 / wt_matx,
                                                                            #            delimiter=",")
                                                                            # np.savetxt("updorweightmatxreal.csv",
                                                                            #            1 / wt_matx_real, delimiter=",")
                                                                            # np.savetxt("updorweightmatxreal1.csv",
                                                                            #            1 / wt_matx_real1, delimiter=",")
                                                                            # print("WFIN")

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

                                                                    break
                                                                k += 1
                                                        else:
                                                            while path_final[k][0] != 0:
                                                                if path_final[k][0] == nodes_real[(node_no, node_links[node_no][next_nodeno])][0].flownumber:
                                                                    path_final[k + 1][2] -= 1
                                                                    path_final[k + 1][9] = (path_final[k + 1][9] + nodes_real[(node_no, node_links[node_no][next_nodeno])][0].total_slot_time) / 2.0
                                                                    Voice_e2e += time_service - nodes_real[(node_no, node_links[node_no][next_nodeno])][0].initial_arrival_time
                                                                    Voice_e2e_Count += 1
                                                                    packetcounter += 1
                                                                    if path_final[k + 1][2] < 1:
                                                                        # print "finished reverse"
                                                                        Voice_Mean_Total = (Voice_Mean_Total + ((path_final[k][9]+path_final[k+1][9]) / 2.0)) / 2.0
                                                                        if True:
                                                                            upde = updateonexit(p, s, d, flow_type, min_rate,
                                                                                                flownumber, userpriority,
                                                                                                path_final[k][0], path_final,
                                                                                                wt_matx,
                                                                                                wt_matx_real,
                                                                                                wt_matx_real1, blockstate)
                                                                            upde.execute()
                                                                            s = upde.s
                                                                            d = upde.d
                                                                            flow_type = upde.flow_type
                                                                            min_rate = upde.min_rate
                                                                            flownumber = upde.flownumber
                                                                            userpriority = upde.userpriority
                                                                            wt_matx = upde.wt_matx
                                                                            wt_matx_real = upde.wt_matx_real
                                                                            wt_matx_real1 = upde.wt_matx_real1
                                                                            path_final = upde.path_final
                                                                            blockstate = upde.blockstate

                                                                            # Debugging
                                                                            # np.savetxt("updweightmatx.csv", 1 / wt_matx,
                                                                            #            delimiter=",")
                                                                            # np.savetxt("updorweightmatxreal.csv",
                                                                            #            1 / wt_matx_real, delimiter=",")
                                                                            # np.savetxt("updorweightmatxreal1.csv",
                                                                            #            1 / wt_matx_real1, delimiter=",")
                                                                            # print("WFIN")

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

                                                                    break
                                                                k += 1
                                                        ################################ Voice Mean Delay Calculations #############
                                                        if Voice_Mean == 0:
                                                            '''
                                                            Voice_Mean = nodes_real[(node_no, node_links[node_no][next_nodeno])][0].service_end_time - \
                                                                         nodes_real[(node_no, node_links[node_no][next_nodeno])][
                                                                             0].initial_arrival_time
                                                            '''
                                                            Voice_Mean = (nodes_real[(node_no, node_links[node_no][next_nodeno])][0].total_slot_time)
                                                        else:
                                                            '''
                                                            Voice_Mean = (
                                                                             Voice_Mean + nodes_real[(node_no, node_links[node_no][next_nodeno])][
                                                                                 0].service_end_time -
                                                                             nodes_real[(node_no, node_links[node_no][next_nodeno])][
                                                                                 0].initial_arrival_time) / 2.0
                                                            '''
                                                            Voice_Mean = (Voice_Mean + nodes_real[(node_no, node_links[node_no][next_nodeno])][
                                                                0].total_slot_time) / 2.0
                                                    else:
                                                        ################### Decrementing the packet count from path_final ############
                                                        k = 0
                                                        if nodes_real[(node_no, node_links[node_no][next_nodeno])][0].direction == True:
                                                            # path_final[nodes_real[(node_no, node_links[node_no][next_nodeno])][0].flownumber ][2] -= 1
                                                            while path_final[k][0] != 0:
                                                                if path_final[k][0] == nodes_real[(node_no, node_links[node_no][next_nodeno])][0].flownumber:
                                                                    path_final[k][2] -= 1
                                                                    path_final[k][9] = (path_final[k][9] + nodes_real[(node_no, node_links[node_no][next_nodeno])][0].total_slot_time) / 2.0
                                                                    packetcounter += 1
                                                                    Video_e2e += time_service - nodes_real[(node_no, node_links[node_no][next_nodeno])][0].initial_arrival_time
                                                                    Video_e2e_Count += 1

                                                                    Video_e2e1 += nodes_real[(node_no, node_links[node_no][next_nodeno])][0].total_slot_time
                                                                    Video_e2e_Count1 += 1
                                                                    if path_final[k][2] < 1:
                                                                        # print "finished"
                                                                        Video_Mean_Total = (Video_Mean_Total + ((path_final[k][9] + path_final[k + 1][9]) / 2.0)) / 2.0
                                                                        if True:
                                                                            upde = updateonexit(p, s, d, flow_type, min_rate,
                                                                                                flownumber, userpriority,
                                                                                                path_final[k][0], path_final,
                                                                                                wt_matx,
                                                                                                wt_matx_real,
                                                                                                wt_matx_real1, blockstate)
                                                                            upde.execute()
                                                                            s = upde.s
                                                                            d = upde.d
                                                                            flow_type = upde.flow_type
                                                                            min_rate = upde.min_rate
                                                                            flownumber = upde.flownumber
                                                                            userpriority = upde.userpriority
                                                                            wt_matx = upde.wt_matx
                                                                            wt_matx_real = upde.wt_matx_real
                                                                            wt_matx_real1 = upde.wt_matx_real1
                                                                            path_final = upde.path_final
                                                                            blockstate = upde.blockstate

                                                                            # Debugging
                                                                            # np.savetxt("updweightmatx.csv", 1 / wt_matx,
                                                                            #            delimiter=",")
                                                                            # np.savetxt("updorweightmatxreal.csv",
                                                                            #            1 / wt_matx_real, delimiter=",")
                                                                            # np.savetxt("updorweightmatxreal1.csv",
                                                                            #            1 / wt_matx_real1, delimiter=",")
                                                                            # print("WFIN")

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

                                                                    break
                                                                k += 1
                                                        else:
                                                            while path_final[k][0] != 0:
                                                                if path_final[k][0] == nodes_real[(node_no, node_links[node_no][next_nodeno])][0].flownumber:
                                                                    path_final[k + 1][2] -= 1
                                                                    path_final[k + 1][9] = (path_final[k + 1][9] + nodes_real[(node_no, node_links[node_no][next_nodeno])][0].total_slot_time) / 2.0
                                                                    Video_e2e += time_service - nodes_real[(node_no, node_links[node_no][next_nodeno])][0].initial_arrival_time
                                                                    Video_e2e_Count += 1
                                                                    packetcounter += 1
                                                                    if path_final[k + 1][2] < 1:
                                                                        # print "finished reverse"
                                                                        Video_Mean_Total = (Video_Mean_Total + ((path_final[k][9] + path_final[k + 1][9]) / 2.0)) / 2.0
                                                                        if True:
                                                                            upde = updateonexit(p, s, d, flow_type, min_rate,
                                                                                                flownumber, userpriority,
                                                                                                path_final[k][0], path_final,
                                                                                                wt_matx,
                                                                                                wt_matx_real,
                                                                                                wt_matx_real1, blockstate)
                                                                            upde.execute()
                                                                            s = upde.s
                                                                            d = upde.d
                                                                            flow_type = upde.flow_type
                                                                            min_rate = upde.min_rate
                                                                            flownumber = upde.flownumber
                                                                            userpriority = upde.userpriority
                                                                            wt_matx = upde.wt_matx
                                                                            wt_matx_real = upde.wt_matx_real
                                                                            wt_matx_real1 = upde.wt_matx_real1
                                                                            path_final = upde.path_final
                                                                            blockstate = upde.blockstate

                                                                            # Debugging
                                                                            # np.savetxt("updweightmatx.csv", 1 / wt_matx,
                                                                            #            delimiter=",")
                                                                            # np.savetxt("updorweightmatxreal.csv",
                                                                            #            1 / wt_matx_real, delimiter=",")
                                                                            # np.savetxt("updorweightmatxreal1.csv",
                                                                            #            1 / wt_matx_real1, delimiter=",")
                                                                            # print("WFIN")

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

                                                                    break
                                                                k += 1
                                                        ################################ Video Mean Delay Calculations #############
                                                        if Video_Mean == 0:
                                                            '''
                                                            Video_Mean = nodes_real[(node_no, node_links[node_no][next_nodeno])][0].service_end_time - \
                                                                         nodes_real[(node_no, node_links[node_no][next_nodeno])][0].initial_arrival_time
                                                            '''
                                                            Video_Mean = nodes_real[(node_no, node_links[node_no][next_nodeno])][0].total_slot_time
                                                        else:
                                                            '''
                                                            Video_Mean = (
                                                                             Video_Mean + nodes_real[(node_no, node_links[node_no][next_nodeno])][
                                                                                 0].service_end_time -
                                                                             nodes_real[(node_no, node_links[node_no][next_nodeno])][
                                                                                 0].initial_arrival_time) / 2.0
                                                            '''
                                                            Video_Mean = (Video_Mean + nodes_real[(node_no, node_links[node_no][next_nodeno])][
                                                                0].total_slot_time) / 2.0
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
                                        # for i in range(0, int(node_service_rate/file_packet_size) - 1, 1):
                                        if dataPacketsAvailable(node_no, node_links[node_no][next_nodeno]):
                                            current_nr_index = 0
                                            current_nr_index = currentNonrealQueue(node_no, node_links[node_no][next_nodeno])
                                            nodes_slot_used[node_no-1][node_links[node_no][next_nodeno]-1] += 1
                                            nodes_slot_total[node_no-1][node_links[node_no][next_nodeno]-1] += 1
                                            arrivaltimes = []
                                            '''
                                            for nd_no in range(1, noOfNodes + 1, 1):
                                                if len(nodes_nonreal[str(nd_no)]) == 0:
                                                    arrivaltimes.append(0)
                                                else:
                                                    arrivaltimes.append(nodes_nonreal[str(nd_no)][0].arrival_time)
                                            '''
                                            #if nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][0].arrival_time <= time_service and B[s_link - 1][d_link - 1] == 8:
                                            s_link = int(nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].path[0])
                                            d_link = int(nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].path[1])
                                            if B[s_link - 1][d_link - 1] == 8:
                                                if nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].flow_tag == 2:
                                                    # initial_service_end = serviceend_time[node_no]
                                                    # if serviceend_time[node_no] == 0:
                                                    #     initial_service_end = 1
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

                                                    # for packetno in range(0, len(nodes_nonreal[(node_no, node_links[node_no][next_nodeno])]), 1):
                                                    #     nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][packetno].addSlotDelay(
                                                    #         file_packet_size / 80000)

                                                    nodes_slot_queue_nonreal_len[node_no-1][node_links[node_no][next_nodeno]-1] += len(nodes_nonreal[(node_no, node_links[node_no][next_nodeno])])
                                                    nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].service(
                                                        max(nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].arrival_time, time_service), B[s_link - 1][d_link - 1],
                                                        False, link_retransmit_prob, file_packet_size / (2000000 / scale))
                                                    # Appending to the serving Queue
                                                    # serviceend_time[node_no] = nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][0].service_end_time
                                                    # packets_tracker[str(node_no)].append(nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][0])
                                                    # Appending to the next node receiving Queue
                                                    if link_retransmit_prob == 1:

                                                        if nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].d_new == 99:  # If packet reached destination we add to the end-to-end final tracker
                                                            ################### Decrementing the packet count from path_final ############
                                                            k = 0
                                                            if nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].direction == True:
                                                                while path_final[k][0] != 0:
                                                                    if path_final[k][0] == nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].flownumber:
                                                                        path_final[k][2] -= 1
                                                                        File_e2e += time_service - nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].initial_arrival_time
                                                                        File_e2e_Count += 1

                                                                        if path_final[k][12] == 1:
                                                                            File_Mean_Total_Time_Count_Satellite += 1
                                                                            File_Mean_Total_Time_Satellite += time_service - nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].initial_arrival_time
                                                                        else:
                                                                            File_Mean_Total_Time_Count_WO_Satellite += 1
                                                                            File_Mean_Total_Time_WO_Satellite += time_service - nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].initial_arrival_time

                                                                        File_Mean_Total_Time_Count_All += 1
                                                                        File_Mean_Total_Time_All += time_service - nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].initial_arrival_time

                                                                        packetcounter += 1
                                                                        '''
                                                                        File_Mean_Speed_e2e = (File_Mean_Speed_e2e + (
                                                                            256.0 / (0.01 * (nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][0].total_slot_time)))) / 2.0
                                                                        '''
                                                                        if path_final[k][2] < 1:
                                                                            sum_soujorn += time_service - path_final[k][5]
                                                                            # if time_service - path_final[k][5] < 5:
                                                                            #     print "pop"
                                                                            number_soujorn += 1
                                                                            # print "finished"

                                                                            if File_Mean_Speed == 0:
                                                                                File_Mean_Speed = (File_Mean_Speed + (
                                                                                path_final[k][7] * path_final[k][
                                                                                    8] * 1000) / (time_service -
                                                                                                  path_final[k][5])) / 1.0
                                                                            else:
                                                                                File_Mean_Speed = (File_Mean_Speed + (
                                                                                path_final[k][7] * path_final[k][
                                                                                    8] * 1000) / (time_service -
                                                                                                  path_final[k][5])) / 2.0
                                                                                '''
                                                                                File_Mean_Speed_e2e = (File_Mean_Speed_e2e + 256.0 / (
                                                                                    0.01 * (
                                                                                        nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][0].service_end_time -
                                                                                        nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][
                                                                                            0].initial_arrival_time))) / 2.0
                                                                                '''

                                                                            upde = updateonexit(p, s, d, flow_type, min_rate,
                                                                                                flownumber, userpriority,
                                                                                                path_final[k][0], path_final,
                                                                                                wt_matx,
                                                                                                wt_matx_real,
                                                                                                wt_matx_real1, blockstate)
                                                                            upde.execute()
                                                                            s = upde.s
                                                                            d = upde.d
                                                                            flow_type = upde.flow_type
                                                                            min_rate = upde.min_rate
                                                                            flownumber = upde.flownumber
                                                                            userpriority = upde.userpriority
                                                                            wt_matx = upde.wt_matx
                                                                            wt_matx_real = upde.wt_matx_real
                                                                            wt_matx_real1 = upde.wt_matx_real1
                                                                            path_final = upde.path_final
                                                                            blockstate = upde.blockstate

                                                                            # Debugging
                                                                            # np.savetxt("updweightmatx.csv", 1 / wt_matx,
                                                                            #            delimiter=",")
                                                                            # np.savetxt("updorweightmatxreal.csv",
                                                                            #            1 / wt_matx_real, delimiter=",")
                                                                            # np.savetxt("updorweightmatxreal1.csv",
                                                                            #            1 / wt_matx_real1, delimiter=",")
                                                                            # print("WFIN")

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

                                                                        break
                                                                    k += 1

                                                        else:
                                                            # nodes_nonreal[str(nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][0].d_new)].append(nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][0])
                                                            # nodes_total_nonreal_packets[nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][0].d_new-1][nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][0].path[1]-1] += 1
                                                            appendNonRealQueue(node_no, node_links[node_no][next_nodeno], current_nr_index)
                                                        nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index].pop(0)
                            time_service = time_service + file_packet_size / 80000
                    if time_service <= min_arrivaltime:
                        for node_no in range(1, noOfNodes + 1, 1):
                            for next_nodeno in range(0, len(node_links[node_no]), 1):
                                if len(nodes_real[(node_no, node_links[node_no][next_nodeno])]) > 0:
                                    nodeoutsidecounter += 1
                                    nodes_slot_used[node_no-1][node_links[node_no][next_nodeno]-1] += 1
                                    nodes_slot_total[node_no-1][node_links[node_no][next_nodeno]-1] += 1
                                    # if nodes_real[(node_no, node_links[node_no][next_nodeno])][0].arrival_time >= max(serviceend_time):
                                    arrivaltimes = []
                                    '''
                                    for nd_no in range(1, noOfNodes + 1, 1):
                                        if len(nodes_real[str(nd_no)]) == 0:
                                            arrivaltimes.append(0)
                                        else:
                                            arrivaltimes.append(nodes_real[str(nd_no)][0].arrival_time)
                                    '''
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
                                            nodes_real[(node_no, node_links[node_no][next_nodeno])][packetno].addSlotDelay(file_packet_size / 20000)

                                        nodes_slot_queue_real_len[node_no-1][node_links[node_no][next_nodeno]-1] += len(nodes_real[(node_no, node_links[node_no][next_nodeno])])

                                        nodes_real[(node_no, node_links[node_no][next_nodeno])][0].service(max(nodes_real[(node_no, node_links[node_no][next_nodeno])][0].arrival_time, time_service),B[s_link - 1][d_link - 1], False, link_retransmit_prob, file_packet_size/20000)
                                        # Appending to the serving Queue
                                        # serviceend_time[node_no] = nodes_real[(node_no, node_links[node_no][next_nodeno])][0].service_end_time
                                        # packets_tracker[str(node_no)].append(nodes_real[(node_no, node_links[node_no][next_nodeno])][0])
                                        if link_retransmit_prob == 1:

                                                # if Voice_Mean_Time[node_no] > 1:
                                                    # print "no"
                                            # Appending to the next node receiving Queue
                                            if nodes_real[(node_no, node_links[node_no][next_nodeno])][0].d_new == 99:
                                                if nodes_real[(node_no, node_links[node_no][next_nodeno])][0].flow_tag == 0:
                                                    ################### Decrementing the packet count from path_final ############
                                                    k = 0
                                                    if nodes_real[(node_no, node_links[node_no][next_nodeno])][0].direction == True:
                                                        # path_final[nodes_real[(node_no, node_links[node_no][next_nodeno])][0].flownumber ][2] -= 1
                                                        while path_final[k][0] != 0:
                                                            if path_final[k][0] == nodes_real[(node_no, node_links[node_no][next_nodeno])][0].flownumber:
                                                                path_final[k][2] -= 1
                                                                path_final[k][9] = (path_final[k][9] + nodes_real[(node_no, node_links[node_no][next_nodeno])][0].total_slot_time) / 2.0
                                                                Voice_e2e += time_service - nodes_real[(node_no, node_links[node_no][next_nodeno])][0].initial_arrival_time
                                                                Voice_e2e_Count += 1
                                                                packetcounter += 1
                                                                if path_final[k][2] < 1:
                                                                    # print "finished"
                                                                    Voice_Mean_Total = (Voice_Mean_Total + ((path_final[k][9] + path_final[k + 1][9]) / 2.0)) / 2.0
                                                                    if True:
                                                                        upde = updateonexit(p, s, d, flow_type, min_rate,
                                                                                            flownumber, userpriority,
                                                                                            path_final[k][0], path_final, wt_matx,
                                                                                            wt_matx_real,
                                                                                            wt_matx_real1, blockstate)
                                                                        upde.execute()
                                                                        s = upde.s
                                                                        d = upde.d
                                                                        flow_type = upde.flow_type
                                                                        min_rate = upde.min_rate
                                                                        flownumber = upde.flownumber
                                                                        userpriority = upde.userpriority
                                                                        wt_matx = upde.wt_matx
                                                                        wt_matx_real = upde.wt_matx_real
                                                                        wt_matx_real1 = upde.wt_matx_real1
                                                                        path_final = upde.path_final
                                                                        blockstate = upde.blockstate

                                                                        # Debugging
                                                                        # np.savetxt("updweightmatx.csv", 1 / wt_matx,
                                                                        #            delimiter=",")
                                                                        # np.savetxt("updorweightmatxreal.csv",
                                                                        #            1 / wt_matx_real, delimiter=",")
                                                                        # np.savetxt("updorweightmatxreal1.csv",
                                                                        #            1 / wt_matx_real1, delimiter=",")
                                                                        # print("WFIN")

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

                                                                break
                                                            k += 1
                                                    else:
                                                        while path_final[k][0] != 0:
                                                            if path_final[k][0] == nodes_real[(node_no, node_links[node_no][next_nodeno])][0].flownumber:
                                                                path_final[k+1][2] -= 1
                                                                path_final[k+1][9] = (path_final[k+1][9] + nodes_real[(node_no, node_links[node_no][next_nodeno])][0].total_slot_time) / 2.0
                                                                Voice_e2e += time_service - nodes_real[(node_no, node_links[node_no][next_nodeno])][0].initial_arrival_time
                                                                Voice_e2e_Count += 1
                                                                packetcounter += 1
                                                                if path_final[k+1][2] < 1:
                                                                    # print "finished reverse"
                                                                    Voice_Mean_Total = (Voice_Mean_Total + ((path_final[k][9] + path_final[k + 1][9]) / 2.0)) / 2.0
                                                                    if True:
                                                                        upde = updateonexit(p, s, d, flow_type, min_rate,
                                                                                            flownumber, userpriority,
                                                                                            path_final[k][0], path_final, wt_matx,
                                                                                            wt_matx_real,
                                                                                            wt_matx_real1, blockstate)
                                                                        upde.execute()
                                                                        s = upde.s
                                                                        d = upde.d
                                                                        flow_type = upde.flow_type
                                                                        min_rate = upde.min_rate
                                                                        flownumber = upde.flownumber
                                                                        userpriority = upde.userpriority
                                                                        wt_matx = upde.wt_matx
                                                                        wt_matx_real = upde.wt_matx_real
                                                                        wt_matx_real1 = upde.wt_matx_real1
                                                                        path_final = upde.path_final
                                                                        blockstate = upde.blockstate

                                                                        # Debugging
                                                                        # np.savetxt("updweightmatx.csv", 1 / wt_matx,
                                                                        #            delimiter=",")
                                                                        # np.savetxt("updorweightmatxreal.csv",
                                                                        #            1 / wt_matx_real, delimiter=",")
                                                                        # np.savetxt("updorweightmatxreal1.csv",
                                                                        #            1 / wt_matx_real1, delimiter=",")
                                                                        # print("WFIN")

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

                                                                break
                                                            k += 1
                                                    ################################ Voice Mean Delay Calculations #############
                                                    if Voice_Mean == 0:
                                                        # Voice_Mean = nodes_real[(node_no, node_links[node_no][next_nodeno])][0].service_end_time - nodes_real[(node_no, node_links[node_no][next_nodeno])][0].initial_arrival_time
                                                        Voice_Mean = nodes_real[(node_no, node_links[node_no][next_nodeno])][0].total_slot_time
                                                    else:
                                                        # Voice_Mean = (Voice_Mean + nodes_real[(node_no, node_links[node_no][next_nodeno])][0].service_end_time - nodes_real[(node_no, node_links[node_no][next_nodeno])][0].initial_arrival_time) / 2.0
                                                        Voice_Mean = (Voice_Mean + nodes_real[(node_no, node_links[node_no][next_nodeno])][0].total_slot_time) / 2.0
                                                else:
                                                    ################### Decrementing the packet count from path_final ############
                                                    k = 0
                                                    if nodes_real[(node_no, node_links[node_no][next_nodeno])][0].direction == True:
                                                        # path_final[nodes_real[(node_no, node_links[node_no][next_nodeno])][0].flownumber ][2] -= 1
                                                        while path_final[k][0] != 0:
                                                            if path_final[k][0] == nodes_real[(node_no, node_links[node_no][next_nodeno])][0].flownumber:
                                                                path_final[k][2] -= 1
                                                                path_final[k][9] = (path_final[k][9] + nodes_real[(node_no, node_links[node_no][next_nodeno])][0].total_slot_time) / 2.0
                                                                Video_e2e += time_service - nodes_real[(node_no, node_links[node_no][next_nodeno])][0].initial_arrival_time
                                                                Video_e2e_Count += 1

                                                                Video_e2e1 += nodes_real[(node_no, node_links[node_no][next_nodeno])][0].total_slot_time
                                                                Video_e2e_Count1 += 1
                                                                packetcounter += 1
                                                                if path_final[k][2] < 1:
                                                                    # print "finished"
                                                                    Video_Mean_Total = (Video_Mean_Total + ((path_final[k][9] + path_final[k + 1][9]) / 2.0)) / 2.0
                                                                    if True:
                                                                        upde = updateonexit(p, s, d, flow_type, min_rate,
                                                                                            flownumber, userpriority,
                                                                                            path_final[k][0], path_final, wt_matx,
                                                                                            wt_matx_real,
                                                                                            wt_matx_real1, blockstate)
                                                                        upde.execute()
                                                                        s = upde.s
                                                                        d = upde.d
                                                                        flow_type = upde.flow_type
                                                                        min_rate = upde.min_rate
                                                                        flownumber = upde.flownumber
                                                                        userpriority = upde.userpriority
                                                                        wt_matx = upde.wt_matx
                                                                        wt_matx_real = upde.wt_matx_real
                                                                        wt_matx_real1 = upde.wt_matx_real1
                                                                        path_final = upde.path_final
                                                                        blockstate = upde.blockstate

                                                                        # Debugging
                                                                        # np.savetxt("updweightmatx.csv", 1 / wt_matx,
                                                                        #            delimiter=",")
                                                                        # np.savetxt("updorweightmatxreal.csv",
                                                                        #            1 / wt_matx_real, delimiter=",")
                                                                        # np.savetxt("updorweightmatxreal1.csv",
                                                                        #            1 / wt_matx_real1, delimiter=",")
                                                                        # print("WFIN")

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

                                                                break
                                                            k += 1
                                                    else:
                                                        while path_final[k][0] != 0:
                                                            if path_final[k][0] == nodes_real[(node_no, node_links[node_no][next_nodeno])][0].flownumber:
                                                                Video_e2e += time_service - nodes_real[(node_no, node_links[node_no][next_nodeno])][0].initial_arrival_time
                                                                Video_e2e_Count += 1
                                                                path_final[k + 1][2] -= 1
                                                                path_final[k+1][9] = (path_final[k+1][9] + nodes_real[(node_no, node_links[node_no][next_nodeno])][0].total_slot_time) / 2.0
                                                                packetcounter += 1
                                                                if path_final[k + 1][2] < 1:
                                                                    # print "finished reverse"
                                                                    Video_Mean_Total = (Video_Mean_Total + ((path_final[k][9] + path_final[k + 1][9]) / 2.0)) / 2.0
                                                                    if True:
                                                                        upde = updateonexit(p, s, d, flow_type, min_rate,
                                                                                            flownumber, userpriority,
                                                                                            path_final[k][0], path_final, wt_matx,
                                                                                            wt_matx_real,
                                                                                            wt_matx_real1, blockstate)
                                                                        upde.execute()
                                                                        s = upde.s
                                                                        d = upde.d
                                                                        flow_type = upde.flow_type
                                                                        min_rate = upde.min_rate
                                                                        flownumber = upde.flownumber
                                                                        userpriority = upde.userpriority
                                                                        wt_matx = upde.wt_matx
                                                                        wt_matx_real = upde.wt_matx_real
                                                                        wt_matx_real1 = upde.wt_matx_real1
                                                                        path_final = upde.path_final
                                                                        blockstate = upde.blockstate

                                                                        # Debugging
                                                                        # np.savetxt("updweightmatx.csv", 1 / wt_matx,
                                                                        #            delimiter=",")
                                                                        # np.savetxt("updorweightmatxreal.csv",
                                                                        #            1 / wt_matx_real, delimiter=",")
                                                                        # np.savetxt("updorweightmatxreal1.csv",
                                                                        #            1 / wt_matx_real1, delimiter=",")
                                                                        # print("WFIN")

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

                                                                break
                                                            k += 1
                                                    ################################ Video Mean Delay Calculations #############
                                                    if Video_Mean == 0:
                                                        '''
                                                        Video_Mean = nodes_real[(node_no, node_links[node_no][next_nodeno])][0].service_end_time - \
                                                                     nodes_real[(node_no, node_links[node_no][next_nodeno])][0].initial_arrival_time
                                                        '''
                                                        Video_Mean = nodes_real[(node_no, node_links[node_no][next_nodeno])][0].total_slot_time
                                                    else:
                                                        '''
                                                        Video_Mean = (
                                                                         Video_Mean + nodes_real[(node_no, node_links[node_no][next_nodeno])][0].service_end_time -
                                                                         nodes_real[(node_no, node_links[node_no][next_nodeno])][0].initial_arrival_time) / 2.0
                                                        '''
                                                        Video_Mean = (Video_Mean + nodes_real[(node_no, node_links[node_no][next_nodeno])][
                                                            0].total_slot_time) / 2.0
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
                                    # for i in range(0, int(node_service_rate/file_packet_size) - 1, 1):
                                    if dataPacketsAvailable(node_no, node_links[node_no][next_nodeno]):
                                        current_nr_index = 0
                                        current_nr_index = currentNonrealQueue(node_no, node_links[node_no][next_nodeno])
                                        nodes_slot_used[node_no-1][node_links[node_no][next_nodeno]-1] += 1
                                        nodes_slot_total[node_no-1][node_links[node_no][next_nodeno]-1] += 1
                                        arrivaltimes = []
                                        '''
                                        for nd_no in range(1, noOfNodes + 1, 1):
                                            if len(nodes_nonreal[str(nd_no)]) == 0:
                                                arrivaltimes.append(0)
                                            else:
                                                arrivaltimes.append(nodes_nonreal[str(nd_no)][0].arrival_time)
                                        '''
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
                                                #     nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][packetno].addSlotDelay(file_packet_size / 20000)

                                                nodes_slot_queue_nonreal_len[node_no-1][node_links[node_no][next_nodeno]-1] += len(nodes_nonreal[(node_no, node_links[node_no][next_nodeno])])

                                                nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].service(max(nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].arrival_time, time_service), B[s_link - 1][d_link - 1], False, link_retransmit_prob, file_packet_size/20000)

                                                # Appending to the serving Queue
                                                # serviceend_time[node_no] = nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][0].service_end_time
                                                # packets_tracker[str(node_no)].append(nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][0])
                                                # Appending to the next node receiving Queue
                                                if link_retransmit_prob == 1:
                                                    if nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].d_new == 99:  # If packet reached destination we add to the end-to-end final tracker
                                                        ################### Decrementing the packet count from path_final ############
                                                        k = 0
                                                        if nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].direction == True:
                                                            while path_final[k][0] != 0:
                                                                if path_final[k][0] == nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].flownumber:
                                                                    path_final[k][2] -= 1
                                                                    File_e2e += time_service - nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].initial_arrival_time
                                                                    File_e2e_Count += 1

                                                                    if path_final[k][12] == 1:
                                                                        File_Mean_Total_Time_Count_Satellite += 1
                                                                        File_Mean_Total_Time_Satellite += time_service - nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].initial_arrival_time
                                                                    else:
                                                                        File_Mean_Total_Time_Count_WO_Satellite += 1
                                                                        File_Mean_Total_Time_WO_Satellite += time_service - nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].initial_arrival_time

                                                                    File_Mean_Total_Time_Count_All += 1
                                                                    File_Mean_Total_Time_All += time_service - nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index][0].initial_arrival_time

                                                                    packetcounter += 1
                                                                    '''
                                                                    File_Mean_Speed_e2e = (File_Mean_Speed_e2e + (
                                                                        256.0 / (0.01 * (nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][0].total_slot_time)))) / 2.0
                                                                    '''
                                                                    if path_final[k][2] < 1:
                                                                        sum_soujorn += time_service - path_final[k][5]
                                                                        # if time_service - path_final[k][5] < 5:
                                                                        #     print "pop"
                                                                        number_soujorn += 1
                                                                        # print "finished"
                                                                        if File_Mean_Speed == 0:
                                                                            File_Mean_Speed = (File_Mean_Speed + (
                                                                            path_final[k][7] * path_final[k][8] * 1000) / (
                                                                                               time_service - path_final[k][
                                                                                                   5])) / 1.0
                                                                        else:
                                                                            File_Mean_Speed = (File_Mean_Speed + (
                                                                            path_final[k][7] * path_final[k][8] * 1000) / (
                                                                                               time_service - path_final[k][
                                                                                                   5])) / 2.0
                                                                        upde = updateonexit(p, s, d, flow_type, min_rate,
                                                                                            flownumber, userpriority,
                                                                                            path_final[k][0], path_final, wt_matx,
                                                                                            wt_matx_real,
                                                                                            wt_matx_real1, blockstate)
                                                                        upde.execute()
                                                                        s = upde.s
                                                                        d = upde.d
                                                                        flow_type = upde.flow_type
                                                                        min_rate = upde.min_rate
                                                                        flownumber = upde.flownumber
                                                                        userpriority = upde.userpriority
                                                                        wt_matx = upde.wt_matx
                                                                        wt_matx_real = upde.wt_matx_real
                                                                        wt_matx_real1 = upde.wt_matx_real1
                                                                        path_final = upde.path_final
                                                                        blockstate = upde.blockstate

                                                                        # Debugging
                                                                        # np.savetxt("updweightmatx.csv", 1 / wt_matx,
                                                                        #            delimiter=",")
                                                                        # np.savetxt("updorweightmatxreal.csv",
                                                                        #            1 / wt_matx_real, delimiter=",")
                                                                        # np.savetxt("updorweightmatxreal1.csv",
                                                                        #            1 / wt_matx_real1, delimiter=",")
                                                                        # print("WFIN")

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

                                                                    break
                                                                k += 1
                                                        ################################ File Mean Delay Calculations #############

                                                    else:
                                                        # nodes_nonreal[str(nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][0].d_new)].append(nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][0])
                                                        # nodes_total_nonreal_packets[nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][0].d_new-1][nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][0].path[1]-1] += 1
                                                        appendNonRealQueue(node_no, node_links[node_no][next_nodeno], current_nr_index)
                                                    nodes_nonreal[(node_no, node_links[node_no][next_nodeno])][current_nr_index].pop(0)
                        time_service = time_service + file_packet_size / 80000
        else:  # Either new flow has arrived or all packets have been served.
            timecurrent = c  # Current Time = Arrival Time
            # print departuretime1, "departuretime1"
            # print flow_type, "flowtype"
            '''
            for loop1 in range(0, countarrival, 1):  # Checking for all calls till last arrival one by one
                # print loop1, "loop1", countarrival - 1
                if departuretime1[loop1] != float('inf'):  # For calls who are yet to depart and already in Network
                    if flowtype[loop1] == 0:  # For Calls which are Real
                        if blockstate1[loop1] == 1:  # For Calls which are not blocked in adapted Dijkstra
                            for loop2 in range(0, 3 * limit - 1, 1):
                                if path_final[loop2, 0] == loop1 + 1:
                                    path1 = path_final[loop2, 5:p + 5]
                                    path2 = path_final[loop2 + 1, 5:p + 5]
                                    break
                            endtoenddelay1 = 0.0
                            endtoenddelay2 = 0.0
                            for loop2 in range(0, p - 1 - 1, 1):
                                if path1[loop2 + 1] != 0:
                                    endtoenddelay1 = endtoenddelay1 + wt_matx_real[int(path1[loop2] - 1), int(
                                        path1[loop2 + 1] - 1)]  # Compute End to End delay of fwd path
                                else:
                                    break
                            for loop2 in range(0, p - 1, 1):
                                if path1[loop2 + 1] != 0:
                                    endtoenddelay2 = endtoenddelay2 + wt_matx_real[int(path1[loop2] - 1), int(
                                        path1[loop2 + 1] - 1)]  # Compute End to End delay of bkwd path
                                else:
                                    break
                            delayvolume[loop1] = delayvolume[loop1] + (endtoenddelay1 + endtoenddelay2) * (
                            timecurrent - timepreviuos1)  # timepreviuos1 = either last arrival or departure
                        if blockstate1_multi[loop1] == 1:  # For Calls which are not blocked in adapted Dijkstra
                            for loop2 in range(0, 3 * limit - 1, 1):
                                if path_final_multi[loop2, 0] == loop1 + 1:
                                    path1 = path_final_multi[loop2, 5:p + 5]
                                    path2 = path_final_multi[loop2 + 1, 5:p + 5]
                                    break
                            endtoenddelay1_multi = 0.0
                            endtoenddelay2_multi = 0.0
                            for loop2 in range(0, p - 1 - 1, 1):
                                if path1[loop2 + 1] != 0:
                                    endtoenddelay1_multi = endtoenddelay1_multi + wt_matx_real_multi[
                                        int(path1[loop2] - 1), int(
                                            path1[loop2 + 1] - 1)]  # Compute End to End delay of fwd path
                                else:
                                    break
                            for loop2 in range(0, p - 1, 1):
                                if path1[loop2 + 1] != 0:
                                    endtoenddelay2_multi = endtoenddelay2_multi + wt_matx_real_multi[
                                        int(path1[loop2] - 1), int(
                                            path1[loop2 + 1] - 1)]  # Compute End to End delay of bkwd path
                                else:
                                    break
                            delayvolume_multi[loop1] = delayvolume_multi[loop1] + (
                                                                                  endtoenddelay1_multi + endtoenddelay2_multi) * (
                                                                                      timecurrent - timepreviuos1)  # timepreviuos1 = either last arrival or departure
                            # print loop1, "loop1end"
            timeprevious1 = timecurrent  # Set as latest arrival data_require
            if countarrival == start + 1:  # Statistics are computed from here
                timeoffirstarrival = flowarrivaltime[I]  # Note the first arrival timecurrent
                timeprevious = flowarrivaltime[
                    I]  # Set timeprevious = current arrival time, because network avg cost calculation starts from here

            if timecurrent > timeprevious:  # Will be false till network avg cost calculation starts

                # Adapted Dijkstra avg network cost computation
                wt_matx1 = wt_matx
                for loop in range(0, p, 1):
                    for loop1 in range(0, p, 1):
                        if wt_matx1[loop, loop1] == float('inf'):
                            wt_matx1[loop, loop1] = 0
                cost = sum(sum(wt_matx1))  # Sum up all link costs
                totalcost = totalcost + cost * (
                timecurrent - timeprevious)  # Update total cost by [cost * (time since last arrival or departure)]
                avgcost = totalcost / (
                timecurrent - timeoffirstarrival)  # Compute avg cost = totalcost/ (time since first arrival when tracking statistics started)
                avgcost1 = np.append(avgcost1, avgcost)
                # End of calculations adapted Dijkstra

                # Multicommodity avg network cost computation
                wt_matx1_multi = wt_matx_multi
                for loop in range(0, p, 1):
                    for loop1 in range(0, p, 1):
                        if wt_matx1_multi[loop, loop1] == float('inf'):
                            wt_matx1_multi[loop, loop1] = 0
                cost_multi = sum(sum(wt_matx1_multi))  # Sum up all link costs
                totalcost_multi = totalcost_multi + cost_multi * (
                timecurrent - timeprevious)  # Update total cost by [cost * (time since last arrival or departure)]
                avgcost_multi = totalcost_multi / (
                    timecurrent - timeoffirstarrival)  # Compute avg cost = totalcost/ (time since first arrival when tracking statistics started)
                avgcost1_multi = np.append(avgcost1_multi, avgcost_multi)
                # End of calculations adapted Dijkstra

                timeprevious = timecurrent  # For next iteration we set the time
            '''
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
            # flowarrivaltime[I] = intialArrival + np.random.exponential(np.divide(1, arrivalrate[I]))
            intialArrival = flowarrivaltime[I]
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
            flownumber_new_multi = flownumber_new_multi + 1
            # Flow number for Enhanced Adapted Dijkstra set to 1 for the first flow
            flownumber_new_block = flownumber_new_block + 1

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
                diff = int((flowarrivaltime[I] - time_service) / ( file_packet_size/ 80000))
                for node_no in range(1, noOfNodes + 1, 1):
                    for next_nodeno in range(0, len(node_links[node_no]), 1):
                        nodes_slot_unused[node_no-1][node_links[node_no][next_nodeno]-1] += diff
                        nodes_slot_total[node_no-1][node_links[node_no][next_nodeno]-1] += diff
                time_service = flowarrivaltime[I]  # If time_service is much lesser than the next flow arrival time , we can fast forward the time as the queue would be empty till flowarrivaltime[I]
            upde = updateonentry1(p, s, d, flow_type, min_rate, flownumber, userpriority, source[I],
                                            destination[I], flow_type1[I], min_rate1[I], flownumber_new,
                                            userpriority_new,
                                            path_final, wt_matx, wt_matx_real, wt_matx_real1, blockstate, flow_duration,
                                            flowarrivaltime[I], connection_type, voice_packet_size, packet_datarate[I],
                                            header_size)
            upde.execute()
            s = upde.s
            d = upde.d
            flow_type = upde.flow_type
            min_rate = upde.min_rate
            flownumber = upde.flownumber
            userpriority = upde.userpriority
            blockstate_new = upde.blockstate_new
            wt_matx = upde.wt_matx
            wt_matx_real = upde.wt_matx_real
            wt_matx_real1 = upde.wt_matx_real1
            path_final = upde.path_final
            blockstate = upde.blockstate

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
            if I <= s6 - 1:
                if I == 0:
                    chosenprob1 = probabilities[0:paths[I]]
                    chosenprob2 = probabilities[3 * total:3 * total + paths[I]]
                else:
                    chosenprob1 = probabilities[paths[I - 1]:paths[I]]
                    chosenprob2 = probabilities[3 * total + paths[I - 1]:3 * total + paths[I]]
            elif I <= 2 * s6 - 1:
                if I == 4:
                    chosenprob1 = probabilities[total:total + paths[I - s6]]
                    chosenprob2 = probabilities[4 * total:4 * total + paths[I - s6]]
                else:
                    chosenprob1 = probabilities[total + paths[I - s6 - 1]:total + paths[I - s6]]
                    chosenprob2 = probabilities[4 * total + paths[I - s6 - 1]:4 * total + paths[I - s6]]
            else:
                if I == 8:
                    chosenprob1 = probabilities[2 * total:2 * total + paths[I - 2 * s6]]
                else:
                    chosenprob1 = probabilities[2 * total + paths[I - 2 * s6 - 1]:2 * total + paths[I - 2 * s6]]
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