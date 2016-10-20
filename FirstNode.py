import random
import csv
from Packets import Packets
import numpy as np

# Network Parameters
node_service_rate = 100000  # Bytes/second
voice_rate = 22000  # Bps
video_rate = 400000  # Bps
file_rate = 300000  # Bps

voice_packet_size = 212  # Bytes
video_packet_size = 1000
file_packet_size = 1000

voice_packet_rate = (voice_rate/voice_packet_size)
video_packet_rate = (video_rate/video_packet_size)
file_packet_rate = (file_rate/file_packet_size)

queue_size = []
queue_time = []
firstarrival_time = 0
firstqueue = 0

'''
flow_tag = 0  -> voice
flow_tag = 1  -> video
flow_tag = 2  -> file
'''

arrival_rate = 0.09
call_duration = 0.09
file_duration = 0.001  # Same for file size as well

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
        firstarrival_time = arrival_rate
        flow_duration = np.random.exponential(np.divide(1, call_duration))
        if flow_duration < 1:
            flow_duration = 1
        flow_tag = np.random.randint(0, 3, size=1)[0]
        print flow_tag, t
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
            flow_duration = np.random.exponential(np.divide(1, file_duration))*1000
            # print "insidetag2", flow_duration/file_packet_size
            if int(flow_duration/file_packet_size) < 1:
                file_limit = 1
            else:
                file_limit = int(flow_duration/file_packet_size)
            for i in range(0, file_limit, 1):
                packets.append(Packets(arrival_time, flow_duration, 2))
                # print "Appending"

        for index in range(0, len(packets) - 1, 1):
            # print index
            if packets[index].flow_tag == 0 or packets[index].flow_tag == 1:
                packets[index].service(packets[0].arrival_time, voice_rate, node_service_rate, True)
                packets1.append(packets[index])
                packets.pop(index)
                break
            else:
                packets[0].service(packets[0].arrival_time, voice_rate, node_service_rate, True)
                packets1.append(packets[0])
                packets.pop(0)
                break
        firstqueue = len(packets1)
    else:
        arrival_time = arrival_time + np.random.exponential(np.divide(1, arrival_rate))
        if (t == 1):
            queue_time.append(arrival_time - firstarrival_time)
        else:
            queue_time.append(arrival_time - queue_time[len(queue_time) - 1])
        flow_duration = np.random.exponential(np.divide(1, call_duration))
        if flow_duration < 1:
            flow_duration = 1
        flow_tag = np.random.randint(0, 3, size=1)[0]
        print flow_tag, t
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
            flow_duration = np.random.exponential(np.divide(1, file_duration))*1000
            # print "insidetag2", flow_duration/file_packet_size
            if int(flow_duration/file_packet_size) < 1:
                file_limit = 1
            else:
                file_limit = int(flow_duration/file_packet_size)
            for i in range(0, file_limit, 1):
                packets.append(Packets(arrival_time, flow_duration, 2))
                # print "Appending"
                # service_start_time = arrival_time
                # packets[i*Int(voice_packet_rate) + j].service(service_start_time, voice_rate, node_service_rate)
                # print packets[i*int(video_packet_rate) + j].arrival_time, flow_tag
    lenght = len(packets)
    # print lenght, "FISKER"
    if lenght != 0:
        index = 0
        i = 0
        flag = 1
        while i < len(packets)-1 and flag == 1:
            # print i, len(packets) - 1
            if packets[i].flow_tag == 0:
                # print len(packets1)
                current_time = packets1[len(packets1) - 1].service_end_time
                if current_time > packets[i].arrival_time:
                    serviceend = packets1[len(packets1) - 1].service_end_time
                    for i in range(0, int(node_service_rate/voice_packet_size) - 1, 1):
                        if i > len(packets) - 1:
                            break
                        packets[i].service(max(packets[i].arrival_time, serviceend), voice_rate, node_service_rate, True)
                        packets1.append(packets[i])
                        packets.pop(i)
                    flag == 0
                    break
            if packets[i].flow_tag == 1:
                # print len(packets1)
                current_time = packets1[len(packets1) - 1].service_end_time
                if current_time > packets[i].arrival_time:
                    serviceend = packets1[len(packets1) - 1].service_end_time
                    for i in range(0, int(node_service_rate/video_packet_size) - 1, 1):
                        if i > len(packets) - 1:
                            break
                        packets[i].service(max(packets[i].arrival_time, serviceend), voice_rate, node_service_rate, True)
                        packets1.append(packets[i])
                        packets.pop(i)
                    flag == 0
                    break
            i = i + 1
        else:
            if len(packets) != 0:
                serviceend = packets1[len(packets1) - 1].service_end_time
                for i in range(0, int(node_service_rate/video_packet_size) - 1, 1):
                    if i > len(packets) - 1:
                        break
                    packets[0].service(max(packets[0].arrival_time, serviceend), voice_rate, node_service_rate, False)
                    packets1.append(packets[0])
                    packets.pop(0)
        lenght = len(packets)
        index = 0
        i = 0
        if t == 0:
            queue_size.append(firstqueue)
        else:
            queue_size.append(len(packets1))
    t = t + 1
lenght = len(packets)
index = 0

Voice_Total_Times = []
Video_Total_Times = []
File_Total_Times = []
Voice_Mean_Time = 0
Video_Mean_Time = 0
File_Mean_Time = 0
for packet in packets1:
    if packet.flow_tag == 0:
        print packet.wait + packet.service_time
        Voice_Total_Times.append(packet.wait + packet.service_time)
    elif packet.flow_tag == 1:
        Video_Total_Times.append(packet.wait+packet.service_time)
    elif packet.flow_tag == 2:
        File_Total_Times.append(packet.wait+packet.service_time)
print Voice_Total_Times
if len(Voice_Total_Times) != 0:
    Voice_Mean_Time = sum(Voice_Total_Times)/len(Voice_Total_Times)
if len(Video_Total_Times) != 0:
    Video_Mean_Time = sum(Video_Total_Times)/len(Video_Total_Times)
if len(File_Total_Times) != 0:
    File_Mean_Time = sum(File_Total_Times)/len(File_Total_Times)
print "Voice Mean Delay: ", Voice_Mean_Time
print "Video Mean Delay: ", Video_Mean_Time
print "File Mean Delay: ", File_Mean_Time
queue_size.pop(len(queue_size) - 1)
queue_sum = np.multiply(queue_size, queue_time)
Mean_Queue_Lenght = sum(queue_sum)/sum(queue_time)
print "Mean Queue Lenght", Mean_Queue_Lenght

if input("Output data to csv (True/False)? "):
    outfile = open('node0.csv', 'wb')
    output = csv.writer(outfile)
    output.writerow(['Customer', 'Arrival_Date', 'Wait', 'Service_Start_Date', 'Service_Time', 'Service_End_Date', 'Flow_type', 'Prioritised'])
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
        outrow.append(packet.prioritised)
        output.writerow(outrow)
    outfile.close()
print ""
