import random
import csv
import numpy as np


class Packets:
    voice_packet_size = 0.025600  # Bytes
    video_packet_size = 0.025600
    file_packet_size = 0.025600

    def __init__(self, initial_arrival_time, arrival_time, flow_duration, flow_tag, path, flownumber, noofpackets, direction, node_service_rate):
        self.initial_arrival_time = initial_arrival_time
        self.arrival_time = arrival_time
        self.pre_arrival = arrival_time
        self.flow_duration = flow_duration
        self.flow_tag = flow_tag
        self.path = path
        self.flownumber = flownumber
        self.s_new = 0
        self.d_new = 0
        self.noofpackets = noofpackets
        self.node_service_rate = node_service_rate
        # True is fwd. False is bkwd
        self.direction = direction
        '''
        self.service_start_date = service_start_date
        self.service_time = service_time
        self.service_end_date = self.service_start_date + self.service_time
        self.wait = self.service_start_date - self.arrival_date
        '''
    def service(self, service_start_time, service_rate, prioritised):
        # self.service_time = (self.service_rate * self.flow_duration)/node_service_rate
        self.pre_arrival = self.arrival_time
        self.s_new = self.path.pop(0)
        if self.path[1] == 0:
            self.d_new = 99
        else:
            self.d_new = self.path[0]
        self.service_start_time = service_start_time
        if self.flow_tag == 0:
            self.service_time = self.voice_packet_size/(self.node_service_rate)
        elif self.flow_tag == 1:
            self.service_time = self.video_packet_size/(self.node_service_rate)
        elif self.flow_tag == 2:
            self.service_time = self.file_packet_size/(self.node_service_rate)
        self.service_end_time = service_start_time + self.service_time
        self.wait = self.service_start_time - self.arrival_time
        # self.arrival_time = self.service_end_time
        self.prioritised = prioritised
    # For Sorting. Will be used for bisect module
    def __lt__(self, other):
        return self.arrival_time < other.arrival_time
