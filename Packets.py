import random
import csv
import numpy as np


class Packets:
    voice_packet_size = 256.00  # Bytes
    video_packet_size = 256.00
    file_packet_size = 256.00


    def __init__(self, initial_arrival_time, arrival_time, flow_duration, flow_tag, path, flownumber,
                 noofpackets, direction, node_service_rate, total_slot_time, total_slots, addedAtSource):
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
        self.total_slot_time = total_slot_time
        self.total_slots = total_slots
        self.file_begin_time = 0
        self.addedAtSource = addedAtSource


    def service(self, service_start_time, service_rate, prioritised, link_retransmit_prob, slot_time):
        # self.service_time = (self.service_rate * self.flow_duration)/node_service_rate
        if link_retransmit_prob == 1:
            self.pre_arrival = self.arrival_time
            self.s_new = self.path.pop(0)
            if len(self.path) == 1:
                self.d_new = 99
            elif self.path[1] == 0:
                self.d_new = 99
            else:
                self.d_new = self.path[0]
        self.service_start_time = service_start_time
        if self.flow_tag == 0:
            self.service_time = slot_time
        elif self.flow_tag == 1:
            self.service_time = slot_time
        elif self.flow_tag == 2:
            if self.file_begin_time == 0:
                self.file_begin_time = service_start_time
            self.service_time = slot_time
        self.service_end_time = service_start_time + self.service_time
        self.wait = self.service_start_time - self.arrival_time
        # self.arrival_time = self.service_end_time
        self.prioritised = prioritised
    # For Sorting. Will be used for bisect module
    def __lt__(self, other):
        return self.arrival_time < other.arrival_time

    def addSlotDelay(self, slot_time):
        self.total_slot_time = self.total_slot_time + slot_time
        self.total_slots = self.total_slots + 1