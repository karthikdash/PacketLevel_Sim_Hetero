import random
import csv
from Packets import Packets
import numpy as np
import bisect

# Network Parameters
node_service_rate = 1000  # Bytes/second
voice_rate = 400  # Bps
video_rate = 400  # Bps
file_rate = 300  # Bps

voice_packet_size = 2.12  # Bytes
video_packet_size = 10.00
file_packet_size = 10.00

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

arrival_rate = 0.001
call_duration = 1.0 / 100
file_duration = 1.0 / 100  # Same for file size as well

print np.random.randint(0, 3, size=1)[0]
print np.random.exponential(np.divide(1, arrival_rate))
print np.random.exponential(call_duration)

# limit = input('Limit:')
limit = 100
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
Voice_Mean_Time0 = 0
Voice_Mean_Time1 = 0
Voice_Mean_Time2 = 0
Voice_Mean_Time3 = 0
Voice_Mean_Time = {
    0: Voice_Mean_Time0,
    1: Voice_Mean_Time1,
    2: Voice_Mean_Time2,
    3: Voice_Mean_Time3
}
Video_Mean_Time0 = 0
Video_Mean_Time1 = 0
Video_Mean_Time2 = 0
Video_Mean_Time3 = 0
Video_Mean_Time = {
    0: Video_Mean_Time0,
    1: Video_Mean_Time1,
    2: Video_Mean_Time2,
    3: Video_Mean_Time3
}
File_Mean_Time0 = 0
File_Mean_Time1 = 0
File_Mean_Time2 = 0
File_Mean_Time3 = 0
File_Mean_Time = {
    0: File_Mean_Time0,
    1: File_Mean_Time1,
    2: File_Mean_Time2,
    3: File_Mean_Time3
}
Voice_Mean = 0
Video_Mean = 0
File_Mean = 0
serviceend_time = [0, 0, 0, 0, 0]
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
        print flow_tag, t, len(packets_realtime0)
        if flow_tag == 0:
            len_tracker = len(packets_realtime0)
            for i in range(1, int(flow_duration), 1):
                for j in range(0, int(voice_packet_rate), 1):
                    bisect.insort_left(packets_realtime0, Packets(arrival_time + float(i + (j)*1.0/voice_packet_rate), arrival_time + float(i + (j)*1.0/voice_packet_rate), flow_duration, flow_tag, path[:], t))
                    # packets_realtime0.append(Packets(arrival_time + float(i + (j)*1.0/voice_packet_rate),arrival_time + float(i + (j)*1.0/voice_packet_rate), flow_duration, flow_tag, path[:], t))
                    # service_start_time = arrival_time
                    # packets[i*Int(voice_packet_rate) + j].service(service_start_time, voice_rate, node_service_rate)
                    # print packets[i*int(voice_packet_rate) + j].arrival_time, flow_tag
            # timetracker.append([packets_realtime[len_tracker-1].arrival_time, len_tracker, len(packets_realtime) - len_tracker - 1, t])
        elif flow_tag == 1:
            len_tracker = len(packets_realtime0)
            for i in range(1, int(flow_duration), 1):
                for j in range(0, int(video_packet_rate), 1):
                    bisect.insort_left(packets_realtime0, Packets(arrival_time + float(i + (j)*1.0/video_packet_rate), arrival_time + float(i + (j)*1.0/video_packet_rate), flow_duration, flow_tag, path[:], t))
                    # packets_realtime0.append(Packets(arrival_time + float(i + (j)*1.0/video_packet_rate),arrival_time + float(i + (j)*1.0/video_packet_rate), flow_duration, flow_tag, path[:], t))
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
                    i = 0
                    initial_service_end = serviceend_time[0]
                    if serviceend_time[0] == 0:
                        initial_service_end = 1
                    while (serviceend_time[node_no] - initial_service_end <= 50):
                        i = i+1
                        if len(nodes_real[str(node_no)]) == 0:
                            break
                        if serviceend_time[0] == 0:
                            nodes_real[str(node_no)][0].service(nodes_real[str(node_no)][0].arrival_time, voice_rate, node_service_rate, False)
                            initial_service_end = nodes_real[str(node_no)][0].arrival_time
                        else:
                            serviceend = serviceend_time[node_no]
                            nodes_real[str(node_no)][0].service(max(nodes_real[str(node_no)][0].arrival_time, serviceend), voice_rate, node_service_rate, False)
                        # Appending to the serving Queue
                        serviceend_time[node_no] = nodes_real[str(node_no)][0].service_end_time
                        # packets_tracker[str(node_no)].append(nodes_real[str(node_no)][0])
                        if nodes_real[str(node_no)][0].flow_tag == 0:
                            Voice_Mean_Time[node_no] = (Voice_Mean_Time[node_no] + nodes_real[str(node_no)][0].service_end_time - nodes_real[str(node_no)][0].arrival_time) / 2.0
                        else:
                            Video_Mean_Time[node_no] = (Video_Mean_Time[node_no] + nodes_real[str(node_no)][0].service_end_time - nodes_real[str(node_no)][0].arrival_time) / 2.0
                        # Appending to the next node receiving Queue
                        if nodes_real[str(node_no)][0].d_new == 99:
                            # final_packets_tracker.append(nodes_real[str(node_no)][0])
                            if nodes_real[str(node_no)][0].flow_tag == 0:
                                if Voice_Mean == 0:
                                    Voice_Mean = nodes_real[str(node_no)][0].service_end_time - nodes_real[str(node_no)][0].initial_arrival_time
                                else:
                                    Voice_Mean = (Voice_Mean + nodes_real[str(node_no)][0].service_end_time - nodes_real[str(node_no)][0].initial_arrival_time) / 2.0
                            else:
                                if Video_Mean == 0:
                                    Video_Mean = nodes_real[str(node_no)][0].service_end_time - nodes_real[str(node_no)][0].initial_arrival_time
                                else:
                                    Video_Mean = (Video_Mean + nodes_real[str(node_no)][0].service_end_time - nodes_real[str(node_no)][0].initial_arrival_time) / 2.0
                        else:
                            # nodes_real[str(nodes_real[str(node_no)][0].d_new)].append(nodes_real[str(node_no)][0])
                            # nodes_real[str(nodes_real[str(node_no)][0].d_new)].append(Packets(nodes_real[str(node_no)][0].initial_arrival_time, nodes_real[str(node_no)][0].service_end_time, nodes_real[str(node_no)][0].flow_duration, nodes_real[str(node_no)][0].flow_tag, nodes_real[str(node_no)][0].path, nodes_real[str(node_no)][0].flownumber))
                            bisect.insort_left(nodes_real[str(nodes_real[str(node_no)][0].d_new)], Packets(nodes_real[str(node_no)][0].initial_arrival_time, nodes_real[str(node_no)][0].service_end_time, nodes_real[str(node_no)][0].flow_duration, nodes_real[str(node_no)][0].flow_tag, nodes_real[str(node_no)][0].path, nodes_real[str(node_no)][0].flownumber))
                        nodes_real[str(node_no)].pop(0)
                        # serviceend = packets_tracker[str(node_no)][len(packets_tracker[str(node_no)]) - 1].service_end_time
            if len(nodes_nonreal[str(node_no)]) != 0 and len(nodes_real[str(node_no)]) == 0:
                # for i in range(0, int(node_service_rate/file_packet_size) - 1, 1):
                initial_service_end = serviceend_time[0]
                if serviceend_time[0] == 0:
                    initial_service_end = 1
                while (serviceend_time[node_no] - initial_service_end <= 0.1):
                    if len(nodes_nonreal[str(node_no)]) == 0:
                        break
                    if serviceend_time[0] == 0:
                        nodes_nonreal[str(node_no)][0].service(nodes_nonreal[str(node_no)][0].arrival_time, voice_rate, node_service_rate, False)
                        initial_service_end = nodes_nonreal[str(node_no)][0].arrival_time
                    else:
                        serviceend = serviceend_time[node_no]
                        nodes_nonreal[str(node_no)][0].service(max(nodes_nonreal[str(node_no)][0].arrival_time, serviceend), voice_rate, node_service_rate, False)
                    # Appending to the serving Queue
                    serviceend_time[node_no] = nodes_nonreal[str(node_no)][0].service_end_time
                    # packets_tracker[str(node_no)].append(nodes_nonreal[str(node_no)][0])
                    # Appending to the next node receiving Queue
                    if nodes_nonreal[str(node_no)][0].d_new == 99:  # If packet reached destination we add to the end-to-end final tracker
                        # final_packets_tracker.append(nodes_nonreal[str(node_no)][0])
                        if File_Mean == 0:
                            File_Mean = nodes_nonreal[str(node_no)][0].service_end_time - nodes_nonreal[str(node_no)][0].initial_arrival_time
                        else:
                            File_Mean = (File_Mean + nodes_nonreal[str(node_no)][0].service_end_time - nodes_nonreal[str(node_no)][0].initial_arrival_time) / 2.0
                    else:
                        # nodes_nonreal[str(nodes_nonreal[str(node_no)][0].d_new)].append(nodes_nonreal[str(node_no)][0])
                        bisect.insort_left(nodes_nonreal[str(nodes_nonreal[str(node_no)][0].d_new)],
                                           Packets(nodes_nonreal[str(node_no)][0].initial_arrival_time,
                                                   nodes_nonreal[str(node_no)][0].service_end_time,
                                                   nodes_nonreal[str(node_no)][0].flow_duration,
                                                   nodes_nonreal[str(node_no)][0].flow_tag,
                                                   nodes_nonreal[str(node_no)][0].path,
                                                   nodes_nonreal[str(node_no)][0].flownumber))
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

Voice_Total_Times0 = []
Voice_Total_Times1 = []
Voice_Total_Times2 = []
Voice_Total_Times = {
    0: Voice_Total_Times0,
    1: Voice_Total_Times1,
    2: Voice_Total_Times2,
}
Video_Total_Times0 = []
Video_Total_Times1 = []
Video_Total_Times2 = []
Video_Total_Times = {
    0: Video_Total_Times0,
    1: Video_Total_Times1,
    2: Video_Total_Times2,
}
File_Total_Times0 = []
File_Total_Times1 = []
File_Total_Times2 = []
File_Total_Times = {
    0: File_Total_Times0,
    1: File_Total_Times1,
    2: File_Total_Times2,
}
packets_tracking = {
    0: packets_tracker0,
    1: packets_tracker1,
    2: packets_tracker2,
    3: final_packets_tracker
}
for i in range(0, 3, 1):
    '''
    for packet in packets_tracking[i]:
        if packet.flow_tag == 0:
            # print packet.wait + packet.service_time
            # Voice_Total_Times.append(packet.wait + packet.service_time)
            Voice_Total_Times[i].append(packet.service_end_time - packet.arrival_time)
        elif packet.flow_tag == 1:
            # Video_Total_Times.append(packet.wait+packet.service_time)
            Video_Total_Times[i].append(packet.service_end_time - packet.arrival_time)
        elif packet.flow_tag == 2:
            # File_Total_Times.append(packet.wait+packet.service_time)
            File_Total_Times[i].append(packet.service_end_time - packet.arrival_time)

    if len(Voice_Total_Times[i]) != 0:
        Voice_Mean_Time[i] = sum(Voice_Total_Times[i])/len(Voice_Total_Times[i])
    if len(Video_Total_Times[i]) != 0:
        Video_Mean_Time[i] = sum(Video_Total_Times[i])/len(Video_Total_Times[i])
    if len(File_Total_Times[i]) != 0:
        File_Mean_Time[i] = sum(File_Total_Times[i])/len(File_Total_Times[i])
    '''
    print "Voice Mean Delay: ", Voice_Mean_Time[i]
    print "Video Mean Delay: ", Video_Mean_Time[i]
    print "File Mean Delay: ", File_Mean_Time[i]
    print ""

'''
Voice_Total_Time = []
Video_Total_Time = []
File_Total_Time = []
for packet in final_packets_tracker:
    if packet.flow_tag == 0:
        # print packet.wait + packet.service_time
        # Voice_Total_Times.append(packet.wait + packet.service_time)
        Voice_Total_Time.append(packet.service_end_time - packet.initial_arrival_time)
    elif packet.flow_tag == 1:
        # Video_Total_Times.append(packet.wait+packet.service_time)
        Video_Total_Time.append(packet.service_end_time - packet.initial_arrival_time)
    elif packet.flow_tag == 2:
        # File_Total_Times.append(packet.wait+packet.service_time)
        File_Total_Time.append(packet.service_end_time - packet.initial_arrival_time)

if len(Voice_Total_Time) != 0:
    Voice_Mean = sum(Voice_Total_Time) / len(Voice_Total_Time)
if len(Video_Total_Time) != 0:
    Video_Mean = sum(Video_Total_Time) / len(Video_Total_Time)
if len(File_Total_Time) != 0:
    File_Mean = sum(File_Total_Times) / len(File_Total_Times)
'''
print "Final Voice Mean Delay: ", Voice_Mean
print "Final Video Mean Delay: ", Video_Mean
print "Final File Mean Delay: ", File_Mean
# queue_size.pop(len(queue_size) - 1)
# queue_sum = np.multiply(queue_size, queue_time)
# Mean_Queue_Lenght = sum(queue_sum)/sum(queue_time)
# print "Mean Queue Lenght", Mean_Queue_Lenght

# if input("Output data to csv (True/False)? "):
if True:
    outfile = open('nodefinal.csv', 'wb')
    output = csv.writer(outfile)
    output.writerow(['Customer', 'Initial_Arrival_Date','Arrival_time','Pre-Arrival', 'Wait', 'delay', 'Service_Start_Date', 'Service_Time', 'Service_End_Date', 'Flow_type', 'Prioritised', 'FlowNumber'])
    i = 0
    for packet in final_packets_tracker:
        i = i+1
        outrow = []
        outrow.append(i)
        outrow.append(packet.initial_arrival_time)
        outrow.append(packet.arrival_time)
        outrow.append(packet.pre_arrival)
        outrow.append(packet.wait)
        outrow.append(packet.service_end_time-packet.arrival_time)
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
            ['Customer', 'Initial_Arrival_Date', 'Arrival_time','Pre_arrival', 'Wait', 'delay', 'Service_Start_Date', 'Service_Time',
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
            outrow.append(packet.service_end_time - packet.arrival_time)
            outrow.append(packet.service_start_time)
            outrow.append(packet.service_time)
            outrow.append(packet.service_end_time)
            outrow.append(packet.flow_tag)
            outrow.append(packet.prioritised)
            outrow.append(packet.flownumber)
            output.writerow(outrow)
        outfile.close()
print ""
