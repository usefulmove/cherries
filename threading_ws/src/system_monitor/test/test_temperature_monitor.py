import unittest
import json
import rclpy
import pickle
import pprint

from threading import Thread

from system_monitor.temperature_node import *

class TestTemperatureMonitor(unittest.TestCase):

    def setUp(self) -> None:

        with open('laptop_temp.pk', 'rb') as fp:
            self.laptop_temp = pickle.load(fp)

        # self.laptop_temp_no_coretemp = self.laptop_temp.copy()
        self.laptop_temp_no_coretemp = {} # empty dict


        # pprint.pp(self.laptop_temp)


        return super().setUp()

    def spin_temperature_monitor(self):

        rclpy.spin(self.temperature_monitor)

        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        self.temperature_monitor.destroy_node()
        rclpy.shutdown()

        
    
    def test_get_intel_cpu(self):
        self.assertEqual(get_intel_cpu(self.laptop_temp), 41.0)
        self.assertEqual(get_intel_cpu(self.laptop_temp_no_coretemp), 0.0)


if __name__ == '__main__':
    unittest.main()