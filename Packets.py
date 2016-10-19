import random
import csv
import numpy as np


class Packets:

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
    def service(self, service_start_time, service_rate, node_service_rate):
        # self.service_time = (self.service_rate * self.flow_duration)/node_service_rate
        self.service_start_time = service_start_time
        self.service_time = np.random.exponential(0.001)
        self.service_end_time = service_start_time + self.service_time
        self.wait = self.service_end_time - self.arrival_time
