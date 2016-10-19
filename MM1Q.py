import random
import csv
from Packets import Packets
import numpy as np

lambd = input('Inter Arrival rate: ')
mu = input('Service rate: ')
simulation_time = input('Total Simulation time:')

t = 0

packets = []
packets1 = []


while t < simulation_time:
    if len(packets) == 0:
        arrival_date = random.expovariate(lambd)
        service_start_date = arrival_date
    else:
        arrival_date = arrival_date + random.expovariate(lambd)
        service_start_date = max(arrival_date, packets[-1].service_end_date)
    service_time = random.expovariate(mu)

    packets.append(Packets(arrival_date, service_start_date, service_time))

    t = arrival_date


for i in range(0, len(packets), 1):
    if i == 0:
        arrival_date = packets[0].service_end_date
        service_start_date = arrival_date
    else:
        arrival_date = packets[i].service_end_date  # Previous packets service end date
        service_start_date = max(arrival_date, packets1[-1].service_end_date)
    service_time = random.expovariate(mu)

    packets1.append(Packets(arrival_date, service_start_date, service_time))


# Calculation
Waits = [packet.wait for packet in packets]
Mean_Wait = sum(Waits)/len(Waits)

Total_times = [packet.wait+packet.service_time for packet in packets]
Mean_Time = sum(Total_times)/len(Total_times)

Service_Times = [packet.service_time for packet in packets]
Mean_Service_Time = sum(Service_Times)/len(Service_Times)

Utilisation = sum(Service_Times)/t
print ""
print "Summary results:"
print ""
print "Number of packets: ", len(packets)
print "Mean Service Time: ", Mean_Service_Time
print "Mean Wait: ", Mean_Wait
print "Mean Time in System: ", Mean_Time
print "Utilisation: ", Utilisation
print ""

if input("Output data to csv (True/False)? "):
    outfile = open('node0.csv', 'wb')
    output = csv.writer(outfile)
    output.writerow(['Customer', 'Arrival_Date', 'Wait', 'Service_Start_Date', 'Service_Time', 'Service_End_Date'])
    i = 0
    for packet in packets:
        i = i+1
        outrow = []
        outrow.append(i)
        outrow.append(packet.arrival_date)
        outrow.append(packet.wait)
        outrow.append(packet.service_start_date)
        outrow.append(packet.service_time)
        outrow.append(packet.service_end_date)
        output.writerow(outrow)
    outfile.close()

    outfile = open('node1.csv', 'wb')
    output = csv.writer(outfile)
    output.writerow(['Customer', 'Arrival_Date', 'Wait', 'Service_Start_Date', 'Service_Time', 'Service_End_Date'])
    i = 0
    for packet in packets1:
        i = i+1
        outrow = []
        outrow.append(i)
        outrow.append(packet.arrival_date)
        outrow.append(packet.wait)
        outrow.append(packet.service_start_date)
        outrow.append(packet.service_time)
        outrow.append(packet.service_end_date)
        output.writerow(outrow)
    outfile.close()
print ""
