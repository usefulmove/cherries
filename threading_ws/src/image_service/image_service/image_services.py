import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
import datetime
import json
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import SetParametersResult

from cherry_interfaces.msg import ImageSetHdr
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.parameter import Parameter

import os.path

from cv_bridge import CvBridge
import cv2

class ImageServices(Node):

    def __init__(self):
        super().__init__('image_services')

        self.br = CvBridge()

        # default parameter values
        self.base_path = '/home/user/Pictures/hdr/'
        base_path_parameter_descriptor = ParameterDescriptor(
            description='Base path where images are to be saved. subfolders will be created with this'
            )
        self.enable_saving = False
        enable_saving_description =ParameterDescriptor(
            description='Turn saving image sets ON (True) or OFF (False).  OFF by default'
            )


        # delcare parameters
        self.declare_parameter('base_path', self.base_path, base_path_parameter_descriptor)
        self.declare_parameter('enable_saving',self.enable_saving, enable_saving_description)

        # check it to make sure it didnt; get set
        self.base_path = self.get_parameter('base_path').get_parameter_value().string_value

        self.add_on_set_parameters_callback(self.parameter_callback)


        self.subscription = self.create_subscription(
            ImageSetHdr,
            'image_set',
            self.save_set_callback,
            10)
        self.subscription  # prevent unused variable warning

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'base_path' :
                try:

                    if (not os.path.exists(param.value)):
                        os.makedirs(param.value)
                    
                    self.base_path = param.value
                except:
                    return SetParametersResult(successful=False, reason = 'Unable to create directory: {}'.format(param.value))
            
            elif param.name == "enable_saving"  :
                try:
                    self.enable_saving = param.value
                except Exception as e:
                    return SetParametersResult(successful=False, reason = 'Error setting parameter: {}'.format(e))
                                
            else:
                return SetParametersResult(successful=False, reason='{} parameter does no exist'.format(param.name))
            
        return SetParametersResult(successful=True)
        

    def save_set_callback(self, imageSetHdr):

        if not self.enable_saving:
            return

        #try:
        self.get_logger().info('Saving images in frame_id : {0}'.format(imageSetHdr.trigger.frame_id))

        now = datetime.datetime.now()
        time_string = now.strftime("%Y%m%dT%H%M%S%f")
        time_date = now.strftime("%Y%m%d")
        time_sstring_nice = now.isoformat(timespec='microseconds')

        #path = self.get_parameter('base_path').get_parameter_value().string_value
        path = os.path.join(self.base_path, time_date, time_string)

        if ( not os.path.exists(path)):
            os.makedirs(path)

        text_name = "{}_{}.json".format(time_string, imageSetHdr.trigger.frame_id)

        text_path = os.path.join(path, text_name)
        self.get_logger().info('data path: {0}'.format(text_path))

        data_dict = {
            'save_time' : time_sstring_nice,
            'ros_time' : {
                'sec' : imageSetHdr.trigger.stamp.sec,
                'nanosec' : imageSetHdr.trigger.stamp.nanosec,
            },
            'frame_id' : imageSetHdr.trigger.frame_id,
            # 'encoder_counts' : {
            #     'bot1' : imageSetHdr.count_bot1,
            #     'bot2' : imageSetHdr.count_bot2,
            #     'top1' : imageSetHdr.count_top1,
            # },
            # 'encoder_mm' : {
            #     'bot1' : imageSetHdr.mm_bot1,
            #     'bot2' : imageSetHdr.mm_bot2,
            #     'top1' : imageSetHdr.mm_top1,
            # },
            # 'image_names' : {
            #     'bot1' : "{}_{}_{}_{}.png".format(time_string, imageSetHdr.frame_id, 'bot', 1),
            #     'bot2' : "{}_{}_{}_{}.png".format(time_string, imageSetHdr.frame_id, 'bot', 2),
            #     'top1' : "{}_{}_{}_{}.png".format(time_string, imageSetHdr.frame_id, 'top', 1),     
            # }
            'images' : [],
        }

        ims = {}

        for im in imageSetHdr.images:
            im_data = {
                'name' : im.name,
                'encoder_count' : im.count,
                'encoder_mm' : im.mm,
                'frame_id' : im.frame_id,
                'image_name' : "{}_{}_{}.png".format(time_string, imageSetHdr.trigger.frame_id, im.name),
            }
            data_dict['images'].append(im_data)

            im_cv2 = self.br.imgmsg_to_cv2(im.image, desired_encoding='passthrough')
            im_path = os.path.join(path,im_data['image_name'])
            cv2.imwrite(im_path, im_cv2)

        with open(text_path, 'w') as convert_file: 
                convert_file.write(json.dumps(data_dict))



def main():
    rclpy.init()

    node = ImageServices()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
