import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
import datetime
import json
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import SetParametersResult

from cherry_interfaces.msg import ImageSetHdr, CherryArray, Cherry
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.parameter import Parameter

import os.path

from cv_bridge import CvBridge
import cv2

class Statistics(Node):

    CHERRY_CLEAN = (1).to_bytes(1, byteorder='big')
    CHERRY_PIT = (2).to_bytes(1, byteorder='big')
    CHERRY_SIDE = (3).to_bytes(1, byteorder='big')
    CHERRY_MAYBE = (5).to_bytes(1, byteorder='big')
    CHERRY_STEM = (6).to_bytes(1, byteorder='big')

    def __init__(self):
        super().__init__('statistics')

        self.totals = {
            'all' : 0,
            Statistics.CHERRY_CLEAN : 0,
            Statistics.CHERRY_PIT : 0,
            Statistics.CHERRY_SIDE : 0,
            Statistics.CHERRY_MAYBE : 0,
            Statistics.CHERRY_STEM : 0,
        }

        self.subscription = self.create_subscription(
            CherryArray,
            'detection_server/detections',
            self.detections_callback,
            10)
        self.subscription  # prevent unused variable warning


    def add_to_totals(self, cherry : Cherry):
        # always increment the all entry
        self.totals['all'] = self.totals['all'] + 1
        # increment the apropriate type in the dict
        if cherry.type in self.totals:
            self.totals[cherry.type] = self.totals[cherry.type] + 1
        else:
            self.totals[cherry.type] = 1

    def log_statistics(self):
        percentages = {
            Statistics.CHERRY_CLEAN : self.totals[Statistics.CHERRY_CLEAN] / self.totals['all'] * 100,
            Statistics.CHERRY_PIT : self.totals[Statistics.CHERRY_PIT] / self.totals['all'] * 100,
            Statistics.CHERRY_SIDE : self.totals[Statistics.CHERRY_SIDE] / self.totals['all'] * 100,
            Statistics.CHERRY_MAYBE : self.totals[Statistics.CHERRY_MAYBE] / self.totals['all'] * 100,
            Statistics.CHERRY_STEM : self.totals[Statistics.CHERRY_STEM] / self.totals['all'] * 100,
        }
        self.get_logger().info('Clean {:.2f}%; Maybe {:.2f}%; Pit {:.2f}%; Side {:.2f}%;'.format(
            percentages[Statistics.CHERRY_CLEAN],
            percentages[Statistics.CHERRY_MAYBE],
            percentages[Statistics.CHERRY_PIT],
            percentages[Statistics.CHERRY_SIDE],
        ))

    def detections_callback(self, msg : CherryArray):
        # self.get_logger().info('got message')
        for cherry in msg.cherries:
            self.add_to_totals(cherry)

        self.log_statistics()


def main():
    rclpy.init()

    node = Statistics()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
