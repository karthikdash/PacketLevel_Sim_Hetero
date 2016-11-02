import random
import csv
from Packets import Packets
import numpy as np
import bisect

# Network Parameters
node_service_rate = 500000  # Bytes/second
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
flow_tag = 2  -> file / non-realtime data
'''

arrival_rate = 0.01
call_duration = 1.0 / 150
file_duration = 1.0 / 150  # Same for file size as well

print np.random.randint(0, 3, size=1)[0]
print np.random.exponential(np.divide(1, arrival_rate))
print np.random.exponential(call_duration)

# limit = input('Limit:')
limit = 20
t = 0

path = [0, 1, 2, 4]

noOfNodes = 4
packets0 = []
packets_realtime0 = []

packets_tracker0 = []
packets_tracker1 = []
packets_tracker2 = []
packets_tracker3 = []
packets_tracker4 = []

packets1 = []
packets_realtime1 = []
packets2 = []
packets_realtime2 = []
packets3 = []
packets_realtime3 = []
packets4 = []
packets_realtime4 = []
nodes_nonreal = {
                '0': packets0,
                '1': packets1,
                '2': packets2,
                '3': packets3,
                '4': packets4
                }
nodes_real = {
                '0': packets_realtime0,
                '1': packets_realtime1,
                '2': packets_realtime2,
                '3': packets_realtime3,
                '4': packets_realtime4
                }
packets_tracker = {
                    '0': packets_tracker0,
                    '1': packets_tracker1,
                    '2': packets_tracker2,
                    '3': packets_tracker3,
                    '4': packets_tracker4
                  }

final_packets_tracker = []
timetracker = []
len_tracker = 0

while t < limit:
    if t == 0:
        arrival_time = np.random.exponential(np.divide(1, arrival_rate))
        firstarrival_time = arrival_rate
        flow_duration = np.random.exponential(np.divide(1, call_duration))
        if flow_duration < 1:
            flow_duration = 1
        flow_tag = np.random.randint(0, 3, size=1)[0]
        # flow_tag = 0
        print flow_tag, t
        if flow_tag == 0:
            for i in range(1, int(flow_duration), 1):
                for j in range(0, int(voice_packet_rate) + 1, 1):
                    packets_realtime0.append(Packets(arrival_time + float(i + (j)*1.0/voice_packet_rate),arrival_time + float(i + (j)*1.0/voice_packet_rate), flow_duration, flow_tag, path[:], t))
                    # service_start_time = arrival_time
                    # packets[i*Int(voice_packet_rate) + j].service(service_start_time, voice_rate, node_service_rate)
                    # print packets[i*int(voice_packet_rate) + j].arrival_time, flow_tag
            # timetracker.append([packets_realtime[len_tracker].arrival_time, len_tracker, len(packets_realtime) - len_tracker - 1, t])
        elif flow_tag == 1:
            for i in range(1, int(flow_duration), 1):
                for j in range(0, int(video_packet_rate), 1):
                    packets_realtime0.append(Packets(arrival_time + float(i + (j)*1.0/video_packet_rate),arrival_time + float(i + (j)*1.0/video_packet_rate), flow_duration, flow_tag, path[:], t))
                    # service_start_time = arrival_time
                    # packets[i*Int(voice_packet_rate) + j].service(service_start_time, voice_rate, node_service_rate)
            # timetracker.append([packets_realtime[len_tracker].arrival_time, len_tracker, len(packets_realtime) - len_tracker - 1, t])
            # print i
        elif flow_tag == 2:
            flow_duration = np.random.exponential(np.divide(1, file_duration))*1000
                    # print packets[i*int(video_packet_rate) + j].arrival_time, flow_tag
            # print "insidetag2", flow_duration/file_packet_size
            if int(flow_duration/file_packet_size) < 1:
                file_limit = 1
            else:
                file_limit = int(flow_duration/file_packet_size)
            for i in range(0, file_limit, 1):
                packets0.append(Packets(arrival_time, arrival_time, flow_duration, 2, path[:], t))
                # print "Appending"
        index = 0
        if len(packets0) != 0 or len(packets_realtime0) != 0:
            # print index
            if len(packets_realtime0) != 0:
                packets_realtime0[0].service(packets_realtime0[0].arrival_time, voice_rate, node_service_rate, False)
                # packets_tracker.append(packets_realtime0[0])
                # Appending to the serving Queue
                packets_tracker0.append(packets_realtime0[0])
                # Appending to the next node receiving queue
                nodes_real[str(packets_realtime0[0].d_new)].append(packets_realtime0[0])
                packets_realtime0.pop(0)
            elif len(packets0) != 0:
                packets0[0].service(packets0[0].arrival_time, file_rate, node_service_rate, False)
                # packets_tracker.append(packets0[0])
                # Appending to the serving Queue
                packets_tracker0.append(packets0[0])
                # Appending to the next node receiving queue
                nodes_nonreal[str(packets0[0].d_new)].append(packets0[0])
                packets0.pop(0)
        firstqueue = len(packets_tracker)
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
        print flow_tag, t, len(packets_tracker)
        if flow_tag == 0:
            len_tracker = len(packets_realtime0)
            for i in range(1, int(flow_duration), 1):
                for j in range(0, int(voice_packet_rate), 1):
                    bisect.insort_left(packets_realtime0, Packets(arrival_time + float(i + (j)*1.0/voice_packet_rate), arrival_time + float(i + (j)*1.0/voice_packet_rate), flow_duration, flow_tag, path[:], t))
                    # service_start_time = arrival_time
                    # packets[i*Int(voice_packet_rate) + j].service(service_start_time, voice_rate, node_service_rate)
                    # print packets[i*int(voice_packet_rate) + j].arrival_time, flow_tag
            # timetracker.append([packets_realtime[len_tracker-1].arrival_time, len_tracker, len(packets_realtime) - len_tracker - 1, t])
        elif flow_tag == 1:
            len_tracker = len(packets_realtime0)
            for i in range(1, int(flow_duration), 1):
                for j in range(0, int(video_packet_rate), 1):
                    bisect.insort_left(packets_realtime0, Packets(arrival_time + float(i + (j)*1.0/video_packet_rate), arrival_time + float(i + (j)*1.0/video_packet_rate), flow_duration, flow_tag, path[:], t))
                    # service_start_time = arrival_time
                    # packets[i*Int(voice_packet_rate) + j].service(service_start_time, voice_rate, node_service_rate)
                    # print packets[i*int(video_packet_rate) + j].arrival_time, flow_tag
            # timetracker.append([packets_realtime[len_tracker-1].arrival_time, len_tracker, len(packets_realtime) - len_tracker - 1, t])
        elif flow_tag == 2:
            flow_duration = np.random.exponential(np.divide(1, file_duration))*1000
            # print "insidetag2", flow_duration/file_packet_size
            if int(flow_duration/file_packet_size) < 1:
                file_limit = 1
            else:
                file_limit = int(flow_duration/file_packet_size)
            for i in range(0, file_limit, 1):
                packets0.append(Packets(arrival_time, arrival_time, flow_duration, 2, path[:], t))
                # print "Appending"
                # service_start_time = arrival_time
                # packets[i*Int(voice_packet_rate) + j].service(service_start_time, voice_rate, node_service_rate)
                # print packets[i*int(video_packet_rate) + j].arrival_time, flow_tag
        lenght = len(packets0)
        # print lenght, "FISKER"

        index = 0
        i = 0
        flag = 1
        # print i, len(packets) - 1
        for node_no in range(0, noOfNodes, 1):
            if len(nodes_real[str(node_no)]) > 0:
                if nodes_real[str(node_no)][0].flow_tag == 0 or nodes_real[str(node_no)][0].flow_tag == 1:
                    # print len(packets_tracker)
                    # for i in range(0, int(node_service_rate/voice_packet_size) - 1, 1):
                    for i in range(0, 1000 - 1, 1):
                        if len(nodes_real[str(node_no)]) == 0:
                            break
                        if len(packets_tracker[str(node_no)]) == 0:
                            nodes_real[str(node_no)][0].service(nodes_real[str(node_no)][0].arrival_time, voice_rate, node_service_rate, False)
                        else:
                            serviceend = packets_tracker[str(node_no)][len(packets_tracker[str(node_no)]) - 1].service_end_time
                            nodes_real[str(node_no)][0].service(max(nodes_real[str(node_no)][0].arrival_time, serviceend), voice_rate, node_service_rate, False)
                        # Appending to the serving Queue
                        packets_tracker[str(node_no)].append(nodes_real[str(node_no)][0])
                        # Appending to the next node receiving Queue
                        if nodes_real[str(node_no)][0].d_new == 99:
                            final_packets_tracker.append(nodes_real[str(node_no)][0])
                        else:
                            nodes_real[str(nodes_real[str(node_no)][0].d_new)].append(nodes_real[str(node_no)][0])
                        nodes_real[str(node_no)].pop(0)
            if len(nodes_nonreal[str(node_no)]) != 0 and len(nodes_real[str(node_no)]) == 0:
                # for i in range(0, int(node_service_rate/file_packet_size) - 1, 1):
                for i in range(0, 100 - 1, 1):
                    if len(nodes_nonreal[str(node_no)]) == 0:
                        break
                    if len(packets_tracker[str(node_no)]) == 0:
                        nodes_nonreal[str(node_no)][0].service(nodes_nonreal[str(node_no)][0].arrival_time, voice_rate, node_service_rate, False)
                    else:
                        serviceend = packets_tracker[str(node_no)][len(packets_tracker[str(node_no)]) - 1].service_end_time
                        nodes_nonreal[str(node_no)][0].service(max(nodes_nonreal[str(node_no)][0].arrival_time, serviceend), voice_rate, node_service_rate, False)
                    # Appending to the serving Queue
                    packets_tracker[str(node_no)].append(nodes_nonreal[str(node_no)][0])
                    # Appending to the next node receiving Queue
                    if nodes_nonreal[str(node_no)][0].d_new == 99:  # If packet reached destination we add to the end-to-end final tracker
                        final_packets_tracker.append(nodes_nonreal[str(node_no)][0])
                    else:
                        nodes_nonreal[str(nodes_nonreal[str(node_no)][0].d_new)].append(nodes_nonreal[str(node_no)][0])
                    nodes_nonreal[str(node_no)].pop(0)

        lenght = len(packets0)
        index = 0
        i = 0
        if t == 0:
            queue_size.append(firstqueue)
        else:
            queue_size.append(len(packets_tracker))
    t = t + 1
lenght = len(packets0)
index = 0

Voice_Total_Times = []
Video_Total_Times = []
File_Total_Times = []
Voice_Mean_Time = 0
Video_Mean_Time = 0
File_Mean_Time = 0
for packet in final_packets_tracker:
    if packet.flow_tag == 0:
        # print packet.wait + packet.service_time
        # Voice_Total_Times.append(packet.wait + packet.service_time)
        Voice_Total_Times.append(packet.service_end_time - packet.initial_arrival_time)
    elif packet.flow_tag == 1:
        # Video_Total_Times.append(packet.wait+packet.service_time)
        Video_Total_Times.append(packet.service_end_time - packet.initial_arrival_time)
    elif packet.flow_tag == 2:
        # File_Total_Times.append(packet.wait+packet.service_time)
        File_Total_Times.append(packet.service_end_time - packet.initial_arrival_time)
# print Voice_Total_Times
if len(Voice_Total_Times) != 0:
    Voice_Mean_Time = sum(Voice_Total_Times)/len(Voice_Total_Times)
if len(Video_Total_Times) != 0:
    Video_Mean_Time = sum(Video_Total_Times)/len(Video_Total_Times)
if len(File_Total_Times) != 0:
    File_Mean_Time = sum(File_Total_Times)/len(File_Total_Times)
print "Voice Mean Delay: ", Voice_Mean_Time
print "Video Mean Delay: ", Video_Mean_Time
print "File Mean Delay: ", File_Mean_Time
# queue_size.pop(len(queue_size) - 1)
# queue_sum = np.multiply(queue_size, queue_time)
# Mean_Queue_Lenght = sum(queue_sum)/sum(queue_time)
# print "Mean Queue Lenght", Mean_Queue_Lenght

if input("Output data to csv (True/False)? "):
    outfile = open('nodefinal.csv', 'wb')
    output = csv.writer(outfile)
    output.writerow(['Customer', 'Initial_Arrival_Date','Arrival_time', 'Wait', 'Service_Start_Date', 'Service_Time', 'Service_End_Date', 'Flow_type', 'Prioritised', 'FlowNumber'])
    i = 0
    for packet in final_packets_tracker:
        i = i+1
        outrow = []
        outrow.append(i)
        outrow.append(packet.initial_arrival_time)
        outrow.append(packet.arrival_time)
        outrow.append(packet.wait)
        outrow.append(packet.service_start_time)
        outrow.append(packet.service_time)
        outrow.append(packet.service_end_time)
        outrow.append(packet.flow_tag)
        outrow.append(packet.prioritised)
        outrow.append(packet.flownumber)
        output.writerow(outrow)
    outfile.close()
    for j in range(0, 3, 1):
        outfile = open(str('node'+str(j)+'.csv'), 'wb')
        output = csv.writer(outfile)
        output.writerow(
            ['Customer', 'Initial_Arrival_Date', 'Arrival_time','Pre_arrival', 'Wait', 'Service_Start_Date', 'Service_Time',
             'Service_End_Date', 'Flow_type', 'Prioritised', 'FlowNumber'])
        i = 0
        for packet in packets_tracker[str(j)]:
            i = i + 1
            outrow = []
            outrow.append(i)
            outrow.append(packet.initial_arrival_time)
            outrow.append(packet.arrival_time)
            outrow.append(packet.pre_arrival)
            outrow.append(packet.wait)
            outrow.append(packet.service_start_time)
            outrow.append(packet.service_time)
            outrow.append(packet.service_end_time)
            outrow.append(packet.flow_tag)
            outrow.append(packet.prioritised)
            outrow.append(packet.flownumber)
            output.writerow(outrow)
        outfile.close()
print ""
