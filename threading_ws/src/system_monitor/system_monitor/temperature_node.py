import psutil

import rclpy
from rclpy.node import Node

import pprint

import pickle

from cherry_interfaces.msg import Temperature

# return 0.0 unless temperature fro 'package ID 0' is found
def get_intel_cpu(temperatures):
    if 'coretemp' in temperatures:
        for entry in temperatures['coretemp']:
            if entry.label == 'Package id 0':
                return entry.current

    return 0.0

def get_nvidia_gpu(temperatures):
    # TODO: figure out how to get this value
    return 0.0

def get_temperatures():
    temperatures = psutil.sensors_temperatures()

    with open('laptop_temp.pk', 'wb') as fp:
        pickle.dump(temperatures, fp)


    # note sure if this is the same across all pc types and so forth
    temp_dict = {
        'cpu' : get_intel_cpu(temperatures),
        'gpu' : get_nvidia_gpu(temperatures),
    }

    return temp_dict

class TemperatureMonitor(Node):

    def __init__(self):
        super().__init__('temperature_monitor')

        self.temperature_publisher = self.create_publisher(
            Temperature, 
            'system/temperature',
            10,
        )

        # publish the temperature once a second
        timer_period = 1.0 # 1 second
        self.timer = self.create_timer(timer_period, self.publish_temperature)

    def publish_temperature(self):

        temps = get_temperatures()
        

        msg = Temperature()
        msg.cpu = temps['cpu']
        msg.gpu = temps['gpu']

        self.temperature_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)

    temp_monitor = TemperatureMonitor()

    rclpy.spin(temp_monitor)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    temp_monitor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
