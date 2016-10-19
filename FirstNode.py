import random
import csv
from Packets import Packets
import numpy as np

# Network Parameters
node_service_rate = 100  # packets/second
voice_rate = 22000  # Bps
video_rate = 400000  # Bps
file_rate = 300000  # Bps

voice_packet_size = 212  # Bytes
video_packet_size = 1000
file_packet_size = 1000

voice_packet_rate = (voice_rate/voice_packet_size)
video_packet_rate = (video_rate/video_packet_size)
file_packet_rate = (file_rate/file_packet_size)

'''
flow_tag = 0  -> voice
flow_tag = 1  -> video
flow_tag = 2  -> file
'''

arrival_rate = 0.001
call_duration = 0.01  # Same for file size as well

print np.random.randint(0, 3, size=1)[0]
print np.random.exponential(np.divide(1, arrival_rate))
print np.random.exponential(call_duration)

limit = input('Limit:')

t = 0

packets = []
packets1 = []


while t < limit:
    if t == 0:
        arrival_time = np.random.exponential(np.divide(1, arrival_rate))
        flow_duration = np.random.exponential(np.divide(1, call_duration))
        flow_tag = np.random.randint(0, 3, size=1)[0]
        print flow_tag
        if flow_tag == 0:
            for i in range(0, int(flow_duration), 1):
                for j in range(0, int(voice_packet_rate), 1):
                    packets.append(Packets(arrival_time + i, flow_duration, flow_tag))
                    # service_start_time = arrival_time
                    # packets[i*Int(voice_packet_rate) + j].service(service_start_time, voice_rate, node_service_rate)
                    # print packets[i*int(voice_packet_rate) + j].arrival_time, flow_tag
        elif flow_tag == 1:
            for i in range(0, int(flow_duration), 1):
                for j in range(0, int(video_packet_rate), 1):
                    packets.append(Packets(arrival_time + i, flow_duration, flow_tag))
                    # service_start_time = arrival_time
                    # packets[i*Int(voice_packet_rate) + j].service(service_start_time, voice_rate, node_service_rate)
                    # print packets[i*int(video_packet_rate) + j].arrival_time, flow_tag
        elif flow_tag == 2:
            for i in range(0, int(flow_duration/video_packet_rate), 1):
                for j in range(0, int(video_packet_rate), 1):
                    packets.append(Packets(arrival_time + i, flow_duration, flow_tag))
                    # service_start_time = arrival_time
                    # packets[i*Int(voice_packet_rate) + j].service(service_start_time, voice_rate, node_service_rate)
                    # print packets[i*int(video_packet_rate) + j].arrival_time, flow_tag
        for index in range(0, len(packets) - 1, 1):
            # print index
            if packets[index].flow_tag == 0 or packets[index].flow_tag == 1:
                packets[index].service(packets[0].arrival_time, voice_rate, node_service_rate)
                packets1.append(packets[index])
                packets.pop(index)
                break
            else:
                packets[0].service(packets[0].arrival_time, voice_rate, node_service_rate)
                packets1.append(packets[0])
                packets.pop(0)
                break
    else:
        arrival_time = arrival_time + np.random.exponential(np.divide(1, arrival_rate))
        flow_duration = np.random.exponential(np.divide(1, call_duration))
        flow_tag = np.random.randint(0, 3, size=1)[0]
        print flow_tag
        if flow_tag == 0:
            for i in range(0, int(flow_duration), 1):
                for j in range(0, int(voice_packet_rate), 1):
                    packets.append(Packets(arrival_time + i, flow_duration, flow_tag))
                    # service_start_time = arrival_time
                    # packets[i*Int(voice_packet_rate) + j].service(service_start_time, voice_rate, node_service_rate)
                    # print packets[i*int(voice_packet_rate) + j].arrival_time, flow_tag
        elif flow_tag == 1:
            for i in range(0, int(flow_duration), 1):
                for j in range(0, int(video_packet_rate), 1):
                    packets.append(Packets(arrival_time + i, flow_duration, flow_tag))
                    # service_start_time = arrival_time
                    # packets[i*Int(voice_packet_rate) + j].service(service_start_time, voice_rate, node_service_rate)
                    # print packets[i*int(video_packet_rate) + j].arrival_time, flow_tag
        elif flow_tag == 2:
            for i in range(0, int(flow_duration/video_packet_rate), 1):
                for j in range(0, int(video_packet_rate), 1):
                    packets.append(Packets(arrival_time + i, flow_duration, flow_tag))
                    # service_start_time = arrival_time
                    # packets[i*Int(voice_packet_rate) + j].service(service_start_time, voice_rate, node_service_rate)
                    # print packets[i*int(video_packet_rate) + j].arrival_time, flow_tag
        for index in range(0, len(packets) - 1, 1):
            if packets[index].flow_tag == 0 or packets[index].flow_tag == 1:
                packets[index].service(max(packets[index].arrival_time, packets1[len(packets1) - 1].service_end_time), voice_rate, node_service_rate)
                packets1.append(packets[index])
                packets.pop(index)
                break
            else:
                packets[0].service(max(packets[index].arrival_time, packets1[len(packets1) - 1].service_end_time), voice_rate, node_service_rate)
                packets1.append(packets[0])
                packets.pop(0)
                break
    t = t + 1
    print t
lenght = len(packets)
index = 0
while index < lenght:
    if (index == lenght):
        break
    print index, lenght
    if packets[index].flow_tag == 0 or packets[index].flow_tag == 1:
        packets[index].service(max(packets[index].arrival_time, packets1[len(packets1) - 1].service_end_time), voice_rate, node_service_rate)
        packets1.append(packets[index])
        packets.pop(index)
    else:
        packets[0].service(max(packets[index].arrival_time, packets1[len(packets1) - 1].service_end_time), voice_rate, node_service_rate)
        packets1.append(packets[0])
        packets.pop(0)
    lenght = len(packets)
    index = 0

if input("Output data to csv (True/False)? "):
    outfile = open('node0.csv', 'wb')
    output = csv.writer(outfile)
    output.writerow(['Customer', 'Arrival_Date', 'Wait', 'Service_Start_Date', 'Service_Time', 'Service_End_Date', 'Flow_type'])
    i = 0
    for packet in packets1:
        i = i+1
        outrow = []
        outrow.append(i)
        outrow.append(packet.arrival_time)
        outrow.append(packet.wait)
        outrow.append(packet.service_start_time)
        outrow.append(packet.service_time)
        outrow.append(packet.service_end_time)
        outrow.append(packet.flow_tag)
        output.writerow(outrow)
    outfile.close()
print ""
