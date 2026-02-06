import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from cherry_interfaces.action import Acquisitionhdr
from cherry_interfaces.msg import EncoderCount
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import os.path

from cv_bridge import CvBridge
import cv2


import json 

class RecordHdr(Node):

    def __init__(self):
        super().__init__('record_hdr')
        self._action_client = ActionClient(self, Acquisitionhdr, 'acquisition')

        self.br = CvBridge()

        self.encoder_sub = self.create_subscription(
            EncoderCount,
            'encoder',
            self.encoder_callback,
            rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value)
        self.encoder_sub  # prevent unused variable warning
        self.encoder_mm = 0

    def send_goal(self, frame_id):
        self.get_logger().info('Sending new request: {}'.format(frame_id))

        goal_msg = Acquisitionhdr.Goal()
        goal_msg.frame_id = frame_id

        self._action_client.wait_for_server()

        self._send_goal_future = self._action_client.send_goal_async(goal_msg)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('Goal handled for frame_id : {0}'.format(result.frame_id))

        path = '/home/user/Pictures/hdr/latest/{}/'.format(result.frame_id)

        if ( not os.path.exists(path)):
            os.makedirs(path)

        text_name = "{}.txt".format(result.frame_id)

        text_path = os.path.join(path, text_name)

        data_dict = {
            'frame_id' : result.frame_id,
            'encoder_counts' : {
                'bot1' : result.count_bot1,
                'bot2' : result.count_bot2,
                'top1' : result.count_top1,
                'top2' : result.count_top2,
            },
            'encoder_mm' : {
                'bot1' : result.mm_bot1,
                'bot2' : result.mm_bot2,
                'top1' : result.mm_top1,
                'top2' : result.mm_top2,
            },
        }

        with open(text_path, 'w') as convert_file: 
             convert_file.write(json.dumps(data_dict))


        image_top1 = self.br.imgmsg_to_cv2(result.image_top1, desired_encoding='passthrough') 
        image_top2 = self.br.imgmsg_to_cv2(result.image_top2, desired_encoding='passthrough') 
        image_bot1 = self.br.imgmsg_to_cv2(result.image_bot1, desired_encoding='passthrough') 
        image_bot2 = self.br.imgmsg_to_cv2(result.image_bot2, desired_encoding='passthrough') 

        self.save_image(image_top1, result.frame_id,"top", 1 )
        self.save_image(image_top2, result.frame_id,"top", 2 )
        self.save_image(image_bot1, result.frame_id,"bot", 1 )
        self.save_image(image_bot2, result.frame_id,"bot", 2 )



    def save_image(self, img, frame_id, location, index):
        name = "{}_{}_{}.bmp".format(frame_id, location, index)
        base = '/home/user/Pictures/hdr/latest/{}'.format(frame_id)

        path = os.path.join(base,name)

        cv2.imwrite(path, img)

    def encoder_callback(self, msg):
        # self.get_logger().info('encoder at : {}'.format(msg.mm))
        current_mm = msg.mm

        if ((current_mm-self.encoder_mm) > 190 ):
            self.encoder_mm = current_mm
            self.send_goal(current_mm)


def main(args=None):
    rclpy.init(args=args)

    action_client = RecordHdr()



    rclpy.spin(action_client)


if __name__ == '__main__':
    main()