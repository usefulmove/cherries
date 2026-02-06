
import queue
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
import datetime
import json
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import SetParametersResult

from cherry_interfaces.msg import ImageSetHdr, CherryArray, Cherry, PickMode, Inputs
from cherry_interfaces.srv import GetCherryBuffer
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, qos_profile_sensor_data
from rclpy.parameter import Parameter

import os.path

from cv_bridge import CvBridge
import cv2




class CherryBuffer(Node):

    CHERRY_CLEAN = (1).to_bytes(1, byteorder='big')
    CHERRY_PIT = (2).to_bytes(1, byteorder='big')
    CHERRY_SIDE = (3).to_bytes(1, byteorder='big')
    CHERRY_MAYBE = (5).to_bytes(1, byteorder='big')
    CHERRY_STEM = (6).to_bytes(1, byteorder='big')

    def __init__(self):
        super().__init__('cherry_buffer')


        # parameters
        self.buffer_length = 10000
        self.mode = 1 #
        self.last_pop = 0
        self.pit_robot_code = 3
        self.stem_robot_code = 4
        self.split_code = 1
        
        # cherry types to send to the robot
        self.to_robot = [
            CherryBuffer.CHERRY_PIT,
        ]

        self.buffers = {
            'pit' : queue.Queue(),
            'clean' : queue.Queue(),
            'maybe' : queue.Queue(),
            'side' : queue.Queue(),
            'stem' : queue.Queue(),
        }

        self.detection_subscription = self.create_subscription(CherryArray, '/detection_server/detections', self.add_to_buffer, 10 )

        self.srv = self.create_service(GetCherryBuffer, '~/get', self.get_callback)

        self.subscription_mode_ = self.create_subscription(PickMode, '~/mode', self.change_mode, 10)

        self.subscription_plc_ = self.create_subscription(Inputs, '/inputs', self.inputs_changed, qos_profile_sensor_data)

    def pop_2(self, reference_mm):
        if self.last_pop == 0:
            cherry_array = self.pop_from_buffer(reference_mm, self.buffers['pit'] )
            code = self.pit_robot_code
            self.last_pop = 1
        else:
            cherry_array = self.pop_from_buffer(reference_mm, self.buffers['stem'] )
            code = self.stem_robot_code
            self.last_pop = 0
    
        return cherry_array, code
    

    def inputs_changed(self, msg : Inputs):

        if (msg.in8):
            if self.mode != 1:
                self.mode = 1
                self.get_logger().info('mode changed to {}'.format(self.mode))
        else:
            if self.mode != 2:
                self.mode = 2
                self.get_logger().info('mode changed to {}'.format(self.mode))
        

    def get_callback(self, request : GetCherryBuffer.Request, response : GetCherryBuffer.Response):

        match self.mode:
            case 1: # send pits to both robots
                response.cherry_array = self.pop_from_buffer(request.reference_mm, self.buffers['pit'] )
                response.model_id = self.split_code
            case 2: # send pits to robot 1 and stems to robot 2
                array, code = self.pop_2(request.reference_mm)
                response.cherry_array = array
                response.model_id = code
        self.get_logger().info('mode : {}, model_id : {}'.format(self.mode, response.model_id))
        # response.cherry_array = self.pop_from_buffer(request.reference_mm )
        return response

    def detection_callback(self, msg : CherryArray):
        self.add_to_buffer(msg)


    def add_to_sub_buffer(self, buffer, cherry, enc_count):
        try:
            buffer.put_nowait((cherry, enc_count))
        except queue.Full:
            # try to pop / push again
            try:
                next_cherry = buffer.get_nowait()
                buffer.put_nowait((cherry, enc_count))
            except:
                # failed a second time?
                pass 

    def add_to_buffer(self, msg : CherryArray):

        # get absolute position for all of the cherries and then push into the queue
        for cherry in msg.cherries:
            type_i = int.from_bytes(cherry.type, "big")
            match type_i:
                case 1:
                    self.add_to_sub_buffer(self.buffers['clean'], cherry, msg.encoder_count)
                case 2:
                    self.add_to_sub_buffer(self.buffers['pit'], cherry, msg.encoder_count)
                case 3:
                    self.add_to_sub_buffer(self.buffers['side'], cherry, msg.encoder_count)
                case 5:
                    self.add_to_sub_buffer(self.buffers['maybe'], cherry, msg.encoder_count)
                case 6:
                    self.add_to_sub_buffer(self.buffers['stem'], cherry, msg.encoder_count)
                case _:
                    self.get_logger().error('unexpected cherry type: {}'.format(type_i))

    def change_mode(self, msg : PickMode):
        if msg.mode == 1 or msg.mode == 2:
            self.mode = msg.mode
        else:
            self.get_logger().error('invalid mode: {}'.format(msg.mode))

    def pop_from_buffer(self, reference_encoder_mm, buffer) -> CherryArray:
        rval = CherryArray()
        rval.encoder_count = reference_encoder_mm
        rval.cherries = []

        while (True):
            try:
                next_cherry, next_mm = buffer.get_nowait()
            except queue.Empty:
                # queue is empty, we are done here
                return rval
            next_cherry.x = (reference_encoder_mm - next_mm)/1000 + next_cherry.x
            if (next_cherry.x < 1000 and next_cherry.x > -1000):
                rval.cherries.append(next_cherry)


def main():
    rclpy.init()

    node = CherryBuffer()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
