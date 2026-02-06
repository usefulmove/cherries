import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cherry_interfaces.msg import ImageSet
from cherry_interfaces.srv import CombineImages

import socket
import sys
import threading
import socket

from cv_bridge import CvBridge
import cv2
import numpy as np
import time







import rclpy
from rclpy.node import Node


class MinimalService(Node):

    def __init__(self):
        super().__init__('image_combiner')
        self.srv = self.create_service(CombineImages, '~/combine', self.combine_callback)

        self.publisher_ = self.create_publisher(Image, '~/combined_image', 10)

        self.scaling = 0.24555903866248 # default value
        self.declare_parameter('pixel_per_count', self.scaling )
        # now we check and see what the value is - it may have been set on startup 
        self.scaling = self.get_parameter('pixel_per_count').get_parameter_value().double_value 
        

    def combine_callback(self, request, response):

        # combine the image
        try:
            combined_image = self.combine_images(request.image_set)

            cv2.imwrite("test.png",combined_image)

            # publish so we can see the image for debuggin pruposes
            image_msg = CvBridge().cv2_to_imgmsg(combined_image, encoding='rgb8')
            self.publisher_.publish(image_msg)

            # return the image
            response.frame_id = request.image_set.frame_id  
            response.image = image_msg

        except Exception as e:

            self.get_logger().error("failed to convert img to msg: {}".format( e))
            
        
        return response

    
    def combine_images(self, image_set):
    #get_logger().info('image one size : {}'.format(image_set.image_top.height, image_set.image_top.width))
    #get_logger().info('image two size: {}'.format(image_set.image_back.height, image_set.image_top.width))

        br = CvBridge()
        # self.scaling = 0.590909090909091

        image_color = None

        try:

            image_top = br.imgmsg_to_cv2(image_set.image_top, desired_encoding='passthrough') 
            image_back = br.imgmsg_to_cv2(image_set.image_bot, desired_encoding='passthrough') 

            # cv2.imwrite("image_top.png",image_top)

            # cv2.imwrite("image_back.png",image_back)

            shape  = image_top.shape
            off_1 = int((image_set.count_top - image_set.count_bot )* self.scaling)


            #if (msg.offset_top != off_):
            #get_logger().info('moving {0}'.format(msg.offset_top ))
            move_matrix_top = np.float32([
                [1, 0, 0],
                [0, 1, float(off_1)]
            ])

            off_2 = 0 # int(image_set.count_bot * self.scaling)

            #if (msg.offset_top != off_):
            #get_logger().info('moving {0}'.format(msg.offset_top ))
            move_matrix_back = np.float32([
                [1, 0, 0],
                [0, 1, float(off_2)]
            ])

            dimensions = (image_top.shape[1], image_top.shape[0])

            #print('step 2 ')
            image_top_moved = cv2.warpAffine(image_top, move_matrix_top, dimensions)
            image_back_moved = cv2.warpAffine(image_back, move_matrix_back, dimensions)



            #image_top_norm = cv2.normalize(temp1, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            zeros = np.zeros((shape), dtype=np.uint8)

            #print('step 3 ')
            image_color = cv2.merge([
                image_back_moved,
                zeros,
                image_top_moved])




        #print(final_img)
        except Exception as e:

            self.get_logger().error("failed to combine image: {}".format( e))
            
            zeros = np.zeros((2464,500,1), dtype=np.uint8)
            image_color = cv2.merge([
                zeros,
                zeros   ,
                zeros])

        return image_color



def main():
    rclpy.init()

    minimal_service = MinimalService()

    rclpy.spin(minimal_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()