import random
import csv
import numpy as np


class Packets:
    voice_packet_size = 212.0  # Bytes
    video_packet_size = 1000.0
    file_packet_size = 1000.0

    def __init__(self, arrival_time, flow_duration, flow_tag):
        self.arrival_time = arrival_time
        self.flow_duration = flow_duration
        self.flow_tag = flow_tag
        '''
        self.service_start_date = service_start_date
        self.service_time = service_time
        self.service_end_date = self.service_start_date + self.service_time
        self.wait = self.service_start_date - self.arrival_date
        '''
    def service(self, service_start_time, service_rate, node_service_rate, prioritised):
        # self.service_time = (self.service_rate * self.flow_duration)/node_service_rate
        self.service_start_time = service_start_time
        if self.flow_tag == 0:
            self.service_time = self.voice_packet_size/node_service_rate
        elif self.flow_tag == 1:
            self.service_time = self.video_packet_size/node_service_rate
        elif self.flow_tag == 2:
            self.service_time = self.file_packet_size/node_service_rate
        self.service_end_time = service_start_time + self.service_time
        self.wait = self.service_start_time - self.arrival_time
        self.prioritised = prioritised
