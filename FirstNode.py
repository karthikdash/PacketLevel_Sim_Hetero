import random
import csv
from Packets import Packets
import numpy as np
import bisect

# Network Parameters
node_service_rate = 5000000  # Bytes/second
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

arrival_rate = 0.01
call_duration = 1.0 / 50
file_duration = 1.0 / 50  # Same for file size as well

print np.random.randint(0, 3, size=1)[0]
print np.random.exponential(np.divide(1, arrival_rate))
print np.random.exponential(call_duration)

# limit = input('Limit:')
limit = 100
t = 0

packets = []
packets_realtime = []
packets1 = []
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
        #flow_tag = 0
        print flow_tag, t
        if flow_tag == 0:
            for i in range(1, int(flow_duration), 1):
                for j in range(0, int(voice_packet_rate) + 1, 1):
                    packets_realtime.append(Packets(arrival_time + float(i + (j)*1.0/voice_packet_rate), flow_duration, flow_tag, t))
                    # service_start_time = arrival_time
                    # packets[i*Int(voice_packet_rate) + j].service(service_start_time, voice_rate, node_service_rate)
                    # print packets[i*int(voice_packet_rate) + j].arrival_time, flow_tag
            timetracker.append([packets_realtime[len_tracker].arrival_time, len_tracker, len(packets_realtime) - len_tracker - 1, t])
        elif flow_tag == 1:
            for i in range(1, int(flow_duration), 1):
                for j in range(0, int(video_packet_rate), 1):
                    packets_realtime.append(Packets(arrival_time + float(i + (j)*1.0/video_packet_rate), flow_duration, flow_tag, t))
                    # service_start_time = arrival_time
                    # packets[i*Int(voice_packet_rate) + j].service(service_start_time, voice_rate, node_service_rate)
                    # print packets[i*int(video_packet_rate) + j].arrival_time, flow_tag
            timetracker.append([packets_realtime[len_tracker].arrival_time, len_tracker, len(packets_realtime) - len_tracker - 1, t])
            print i
        elif flow_tag == 2:
            flow_duration = np.random.exponential(np.divide(1, file_duration))*1000
            # print "insidetag2", flow_duration/file_packet_size
            if int(flow_duration/file_packet_size) < 1:
                file_limit = 1
            else:
                file_limit = int(flow_duration/file_packet_size)
            for i in range(0, file_limit, 1):
                packets.append(Packets(arrival_time, flow_duration, 2, t))
                # print "Appending"
        index = 0
        while len(packets) != 0 or len(packets_realtime) != 0:
            # print index
            if len(packets_realtime) != 0:
                packets_realtime[0].service(packets_realtime[0].arrival_time, voice_rate, node_service_rate, False)
                packets1.append(packets_realtime[0])
                packets_realtime.pop(index)
                for tracker in list(timetracker):
                    if tracker[3] == packets_realtime[0].flownumber:
                        tracker[1] = tracker[1] + 1
                        tracker[2] = tracker[2] - 1
                break
            elif len(packets) != 0:
                packets[0].service(packets[0].arrival_time, file_rate, node_service_rate, False)
                packets1.append(packets[0])
                packets.pop(0)
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
        print flow_tag, t, len(packets1)
        if flow_tag == 0:
            len_tracker =  len(packets_realtime)
            for i in range(1, int(flow_duration), 1):
                for j in range(0, int(voice_packet_rate), 1):
                    packets_realtime.append(Packets(arrival_time + float(i + (j)*1.0/voice_packet_rate), flow_duration, flow_tag, t))
                    # service_start_time = arrival_time
                    # packets[i*Int(voice_packet_rate) + j].service(service_start_time, voice_rate, node_service_rate)
                    # print packets[i*int(voice_packet_rate) + j].arrival_time, flow_tag
            timetracker.append([packets_realtime[len_tracker-1].arrival_time, len_tracker, len(packets_realtime) - len_tracker - 1, t])
        elif flow_tag == 1:
            len_tracker = len(packets_realtime)
            for i in range(1, int(flow_duration), 1):
                for j in range(0, int(video_packet_rate), 1):
                    packets_realtime.append(Packets(arrival_time + float(i + (j)*1.0/video_packet_rate), flow_duration, flow_tag, t))
                    # service_start_time = arrival_time
                    # packets[i*Int(voice_packet_rate) + j].service(service_start_time, voice_rate, node_service_rate)
                    # print packets[i*int(video_packet_rate) + j].arrival_time, flow_tag
            timetracker.append([packets_realtime[len_tracker-1].arrival_time, len_tracker, len(packets_realtime) - len_tracker - 1, t])
        elif flow_tag == 2:
            flow_duration = np.random.exponential(np.divide(1, file_duration))*1000
            # print "insidetag2", flow_duration/file_packet_size
            if int(flow_duration/file_packet_size) < 1:
                file_limit = 1
            else:
                file_limit = int(flow_duration/file_packet_size)
            for i in range(0, file_limit, 1):
                packets.append(Packets(arrival_time, flow_duration, 2, t))
                # print "Appending"
                # service_start_time = arrival_time
                # packets[i*Int(voice_packet_rate) + j].service(service_start_time, voice_rate, node_service_rate)
                # print packets[i*int(video_packet_rate) + j].arrival_time, flow_tag
        lenght = len(packets)
        # print lenght, "FISKER"

        index = 0
        i = 0
        flag = 1
        if (len(packets)>0 or len(packets_realtime)>0):
            # print i, len(packets) - 1
            if(len(packets_realtime) > 0):
                if packets_realtime[0].flow_tag == 0:
                    # print len(packets1)
                    serviceend = packets1[len(packets1) - 1].service_end_time
                    # for i in range(0, int(node_service_rate/voice_packet_size) - 1, 1):
                    for i in range(0, 100 - 1, 1):
                        if len(packets_realtime) == 0:
                            break
                        packets_realtime[0].service(max(packets_realtime[0].arrival_time, serviceend), voice_rate, node_service_rate, False)
                        packets1.append(packets_realtime[0])
                        for tracker in timetracker:
                            if tracker[3] != packets_realtime[0].flownumber and tracker[2] != -1:
                                if tracker[0] < serviceend:
                                    print "k"
                                    serviceend = packets1[len(packets1) - 1].service_end_time
                                    packets_realtime[tracker[1]].service(max(packets_realtime[tracker[1]].arrival_time, serviceend), voice_rate, node_service_rate, False)
                                    packets1.append(tracker[1])
                                    packets_realtime.pop(tracker[1])
                                    # Tracking Updation
                                    tracker[0] = packets_realtime[tracker[1] + 1].arrival_time
                                    tracker[1] = tracker[1] + 1
                                    tracker[2] = tracker[2] - 1
                        if len(packets_realtime) == 1:
                            # If all the realtime packets of a particular flow have been serviced, we have to remove that flow from time tracker
                            # Why: This is no longer needed in checking if flows arrived at currentime of the system
                            for tracker in timetracker:
                                if tracker[3] == packets_realtime[0].flownumber:
                                    timetracker.remove(tracker)
                        for tracker in list(timetracker):
                            if tracker[3] == packets_realtime[0].flownumber:
                                tracker[0] = packets_realtime[0].arrival_time
                                tracker[1] = tracker[1] + 1
                                tracker[2] = tracker[2] - 1
                        packets_realtime.pop(0)
                '''
                if len(packets_realtime) > 0:
                    if packets_realtime[0].flow_tag == 1:
                        # print len(packets1)
                        serviceend = packets1[len(packets1) - 1].service_end_time
                        # for i in range(0, int(node_service_rate/video_packet_size) - 1, 1):
                        for i in range(0, 100 - 1, 1):
                            if len(packets_realtime) == 0:
                                break
                            packets_realtime[0].service(max(packets_realtime[0].arrival_time, serviceend), video_rate, node_service_rate, False)
                            packets1.append(packets_realtime[0])
                            for tracker in timetracker:
                                if tracker[3] != packets_realtime[0].flownumber:
                                     if tracker[0] < serviceend:
                                        print "k"
                            if len(packets_realtime) == 1:
                                for tracker in timetracker:
                                    if tracker[3] == packets_realtime[0].flownumber:
                                        timetracker.remove(tracker)
                            for tracker in list(timetracker):
                                if tracker[3] == packets_realtime[0].flownumber:
                                    tracker[0] = packets_realtime[0].arrival_time
                                    tracker[1] = tracker[1] + 1
                                    tracker[2] = tracker[2] - 1
                            packets_realtime.pop(0)
                '''
                i = i + 1
            if len(packets) != 0 and len(packets_realtime) == 0:
                serviceend = packets1[len(packets1) - 1].service_end_time
                # for i in range(0, int(node_service_rate/file_packet_size) - 1, 1):
                for i in range(0, 50 - 1, 1):
                    if len(packets) == 0:
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
        # print packet.wait + packet.service_time
        Voice_Total_Times.append(packet.wait + packet.service_time)
    elif packet.flow_tag == 1:
        Video_Total_Times.append(packet.wait+packet.service_time)
    elif packet.flow_tag == 2:
        File_Total_Times.append(packet.wait+packet.service_time)
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
    outfile = open('node0.csv', 'wb')
    output = csv.writer(outfile)
    output.writerow(['Customer', 'Arrival_Date', 'Wait', 'Service_Start_Date', 'Service_Time', 'Service_End_Date', 'Flow_type', 'Prioritised', 'FlowNumber'])
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
        outrow.append(packet.flownumber)
        output.writerow(outrow)
    outfile.close()
print ""
