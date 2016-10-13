import numpy as np
from updateonentry1 import updateonentry1
from udpateonentry import updateonentry
# Network Parameters

# Total Number of Nodes
n = 10
p = n


# Link is (link_src[i],link_dest[i])
link_src = [1, 1, 2, 2, 3, 4, 4, 5, 5, 5, 6, 7, 8, 9]
link_dest = [2, 8, 3, 9, 4, 5, 10, 6, 7, 10, 7, 8, 10, 10]
link_dest12 = [2, 8, 3, 9, 4, 5, 9, 6, 7, 9, 7, 8, 9, 9]


# P[i][j],1
link_onprob1 = [0.3, 0.3, 0.2, 0.2, 0.2, 0.3, 0.2, 0.3, 0.2, 0.2, 0.3, 0.2, 0.2, 0.3]
# E[i][j],1
link_errorprob1 = [0.07, 0.08, 0.07, 0.07, 0.07, 0.08, 0.07, 0.08, 0.07, 0.07, 0.08, 0.07, 0.07, 0.07]
# P[i][j],2
link_onprob2 = [0.2, 0.2, 0.3, 0.2, 0.3, 0.3, 0.3, 0.2, 0.2, 0.3, 0.2, 0.2, 0.3, 0.2]
# E[i][j],2
link_errorprob2 = [0.05, 0.06, 0.04, 0.05, 0.04, 0.06, 0.05, 0.05, 0.05, 0.04, 0.06, 0.06, 0.04, 0.05]
# P[i][j],3
link_onprob3 = [0.2, 0.2, 0.2, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.2, 0.2]
# E[i][j],3
link_errorprob3 = [0.01, 0.02, 0.01, 0.02, 0.02, 0.03, 0.03, 0.01, 0.02, 0.02, 0.03, 0.02, 0.01, 0.01]

# (Half Bandwidth/frameSize * [Linke Rates corresponding to link_src[i],link_dest[j]]
link_rate = np.multiply((1000000.0/256), [2, 2, 8, 8, 2, 8, 2, 8, 8, 2, 2, 2, 8, 2])

# <M> Defined in source-destination pairs with rate requirements
source1 = [2, 4, 3, 1]
destination1 = [6, 8, 7, 5]
s5 = 1
s6 = len(source1)

# Service Time is exponentially distributed with mean T
T = 150
# Arrival Rate
lamb = 0.001

# <M> Data Rate Requirements
data_require = [22, 80, 22, 11, 400, 400, 400, 400, 300, 400, 300, 300]
# 232 = frame size - overheads size
min_rate1 = np.multiply(1000.0/232, data_require)
min_rate2 = np.multiply(T*lamb*(1000.0/232), data_require)
flow_type1 = [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2]
arrivalrate = np.multiply(0.001, np.ones((12)))
servicetime = np.multiply(150, np.ones((12)))
# Video,Voice and Realtime?
connectiontypes = 3

# Iterations (Higher value can lead to long execution times)
# limit = 100000
limit = 500
# Observation from 'start' iteration ?
start = 200
# Probability with which blocked call will be retried
retry_prob = 0.7
# Sigma
weight_x = ((np.multiply(np.array(link_onprob1), np.subtract(1, link_errorprob1))) +
            (np.multiply(np.array(link_onprob2), np.subtract(1, link_errorprob2))) +
            (np.multiply(np.array(link_onprob3), np.subtract(1, link_errorprob3))))
weight = np.multiply(weight_x, link_rate)

A = np.zeros((n, n))
m1 = np.shape(link_src)[0]
# Matlab indexing starts from 1 but Python starts from 0
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
np.savetxt("foo3.csv", a, delimiter=',')

# To ignore Division by O warnings. /0 taken as Inf
with np.errstate(divide='ignore', invalid='ignore'):
    wt_matx = np.divide(1, eff_capacity_matx)
    wt_matx_real = np.divide(1, eff_capacity_matx)
    wt_matx_real1 = np.divide(2, eff_capacity_matx)
    wt_matx_multi = np.divide(1, eff_capacity_matx)
    wt_matx_real_multi = np.divide(1, eff_capacity_matx)
    wt_matx_real1_multi = np.divide(2, eff_capacity_matx)
    wt_matx_block = np.divide(1, eff_capacity_matx)
    wt_matx_real_block = np.divide(1, eff_capacity_matx)
    wt_matx_real1_block = np.divide(2, eff_capacity_matx)
source = []
destination = []
for i in range(0, connectiontypes):
    source = np.append(source, source1)
    destination = np.append(destination, destination1)
# ##
s = []
d = []
flow_type = []
# Not sure
# min_rate = []
min_rate = 0
flownumber = []
userpriority = []
blockstate = []
userpriority_new = 1
flownumber_new = 0

# ##
s_multi = []
d_multi = []
flow_type_multi = []
min_rate_block = []
flownumber_multi = []
userpriority_multi = []
blockstate_multi = []
userpriority_multi = 1
flownumber_new_multi = 0

# ##
s_block = []
d_block = []
flow_type_block = []
flownumber_block = []
userpriority_block = []
blockstate_block = []
userpriority_new_block = 1
flownumber_new_block = 0

# ##
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
path_final = np.zeros((3*limit, p+5))
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

# Input Parameters for Multicommodity Flow #

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
for i in range(0, m1, 1):
    probabilities[i] = fractions[i]/min_rate2[0]
    probabilities[total + i] = fractions[total + i]/min_rate2[s6 + 0]
    probabilities[2*total + i] = fractions[2*total + i]/min_rate2[2*s6 + 0]
    probabilities[3*total + i] = fractions[3*total + i]/min_rate2[0]
    probabilities[4*total + i] = fractions[4*total + i]/min_rate2[s6 + 0]
for i in range(0, m2, 1):
    probabilities[m1 + i] = fractions[m1 + i]/min_rate2[1]
    probabilities[total + m1 + i] = fractions[total + m1 + i]/min_rate2[s6 + 1]
    probabilities[2*total + m1 + i] = fractions[2*total + m1 + i]/min_rate2[2*s6 + 1]
    probabilities[3*total + m1 + i] = fractions[3*total + m1 + i]/min_rate2[1]
    probabilities[4*total + m1 + i] = fractions[4*total + m1 + i]/min_rate2[s6 + 1]
for i in range(0, m3, 1):
    probabilities[m1 + m2 + i] = fractions[m1 + i]/min_rate2[2]
    probabilities[total + m1 + m2 + i] = fractions[total + m1 + m2 + i]/min_rate2[s6 + 2]
    probabilities[2*total + m1 + m2 + i] = fractions[2*total + m1 + m2 + i]/min_rate2[2*s6 + 2]
    probabilities[3*total + m1 + m2 + i] = fractions[3*total + m1 + m2 + i]/min_rate2[2]
    probabilities[4*total + m1 + m2 + i] = fractions[4*total + m1 + m2 + i]/min_rate2[s6 + 2]
for i in range(0, m4, 1):
    probabilities[m1 + m2 + m3 + i] = fractions[m1 + m2 + m3 + i]/min_rate2[3]
    probabilities[total + m1 + m2 + m3 + i] = fractions[total + m1 + m2 + m3 + i]/min_rate2[s6 + 3]
    probabilities[2*total + m1 + m2 + m3 + i] = fractions[2*total + m1 + m2 + m3 + i]/min_rate2[2*s6 + 3]
    probabilities[3*total + m1 + m2 + m3 + i] = fractions[3*total + m1 + m2 + m3 + i]/min_rate2[3]
    probabilities[4*total + m1 + m2 + m3 + i] = fractions[4*total + m1 + m2 + m3 + i]/min_rate2[s6 + 3]

# ##
arrivaltime = []
departuretime = []
departuretime1 = []

# Exponential Randome distribution at mean 1/lambda
flowarrivaltime = np.random.exponential(np.divide(1, arrivalrate))
arrivalratesize = np.shape(arrivalrate)[0]
arrivalratesize1 = 1
print flowarrivaltime.min()
print flowarrivaltime.argmin()
# Flow starts
# while(countarrival < limit):
while(countarrival < 1):
    c = flowarrivaltime.min()  # Minimum Value
    I = flowarrivaltime.argmin()  # Index of the Minimum Value
    if countarrival == 0:
        countarrival = countarrival + 1
        timeprevious = flowarrivaltime[I]  # First Flow arrival time
        # Arrivaltime vector is updated by appending the current flow arrival time which just arrived
        arrivaltime = np.append(arrivaltime, flowarrivaltime[I])
        # <PDQ> Time of departure udpated for the flow that as arrived
        timeofdeparture = c + np.random.exponential(servicetime[I])
        # DepartureTime vector is updated by appending the current flow departure time which just arrived
        departuretime = np.append(departuretime, timeofdeparture)
        departuretime1 = np.append(departuretime1, timeofdeparture)
        # New Arrival time computation for the next flow
        flowarrivaltime[I] = flowarrivaltime[I] + np.random.exponential(np.divide(1, arrivalrate[I]))
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
        ################################################
        # updateonentry1 does routing using Adapted Dijkstra
        ################################################

        updateonentry1 = updateonentry1(p, s, d, flow_type, min_rate, flownumber, userpriority, source[I],
                                        destination[I], flow_type1[I], min_rate1[I], flownumber_new, userpriority_new,
                                        path_final, wt_matx, wt_matx_real, wt_matx_real1, blockstate)
        updateonentry1.execute()
        updateonentry1.wt_matx_real1
        if updateonentry1.blockstate_new == 0:  # If call is blocked by apadted Dijkstra
            count_algo1 = count_algo1 + 1
        blockstate1[countarrival] = updateonentry1.blockstate_new
        ################################################
        # updateonentry1 does routing using Enhanced Dijkstra
        ################################################

        updateonentry = updateonentry(p, s_block, d_block, flow_type_block, min_rate_block, flownumber_block, userpriority_block, source[I], destination[I], flow_type1[I], min_rate1[I], flownumber_new_block, userpriority_new_block, path_final_block, wt_matx_block, wt_matx_real_block, wt_matx_real1_block, eff_capacity_matx, blockstate_block)

    else:
