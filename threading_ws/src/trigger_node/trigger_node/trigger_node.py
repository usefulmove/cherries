import rclpy
from rclpy.node import Node

import os

from cherry_interfaces.msg import Trigger
from cherry_interfaces.msg import EncoderCount
import rclpy.time

from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import SetParametersResult

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

class TriggerNode(Node):

    def __init__(self):
        super().__init__('trigger_node')

        self.current_mm = 0
        self.last_mm = 0
        self.frame_id = 0
        self.trigger_interval = 115 # mm

        trigger_interval_descriptor = ParameterDescriptor(
            description='Distance between camera triggers in mm.'
            )
        self.declare_parameter('trigger_interval', self.trigger_interval, trigger_interval_descriptor)
        self.trigger_interval = self.get_parameter('trigger_interval').get_parameter_value().integer_value
        self.add_on_set_parameters_callback(self.parameter_callback)

        self.trigger_publisher = self.create_publisher(
            Trigger, 
            'trigger',
            10,

        
        )

        # monitor encoder to check and see if we are receiving reasonably spaced
        # triggers
        plc_io_qos = rclpy.qos.qos_profile_sensor_data
        plc_io_qos.lifespan = rclpy.qos.Duration(seconds=1)
        plc_io_qos.deadline = rclpy.qos.Duration(seconds=1)

        self.encoder_subscriber = self.create_subscription(
            EncoderCount, 
            'encoder', 
            self.encoder_callback,
            plc_io_qos
        )



    def parameter_callback(self, params):
        for param in params:
            if param.name == 'trigger_interval' :
                try:
                    self.trigger_interval = param.value
                except Exception as e:
                    return SetParametersResult(successful=False, reason = 'Unable to set trigger_interval: {}'.format(e))              
            else:
                return SetParametersResult(successful=False, reason='{} parameter does no exist'.format(param.name))
            
        return SetParametersResult(successful=True)
        

    def trigger(self, encoder_mm, encoder_count):
        self.frame_id = self.frame_id + 1

        self.get_logger().info('Trigger Frame {} at {} mm.'.format(self.frame_id, encoder_mm))
        msg = Trigger()
        msg.frame_id = self.frame_id
        msg.encoder_mm = encoder_mm
        msg.encoder_count = encoder_count
        msg.stamp = self.get_clock().now().to_msg()

        self.trigger_publisher.publish(msg)


    def encoder_callback(self, encoder_message):
        self.current_mm = encoder_message.mm

        if (self.last_mm > self.current_mm):
            # we are goin backwards?
            # or perhaps the encoder value rolled over
            # make sure last is always smaller than current
            self.last_mm = self.current_mm 

        if ((self.current_mm - self.last_mm) >= self.trigger_interval):
            self.last_mm = int(self.current_mm  / self.trigger_interval) * self.trigger_interval
            self.trigger(encoder_message.mm, encoder_message.count)



def main(args=None):
    rclpy.init(args=args)

    temp_monitor = TriggerNode()

    rclpy.spin(temp_monitor)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    temp_monitor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
