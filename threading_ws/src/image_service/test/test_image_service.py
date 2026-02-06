import unittest
import cv2
import json
from cherry_interfaces.msg import ImageSetHdr, ImageLayer
from cv_bridge import CvBridge
import rclpy
from image_service.image_services import ImageServices
import rclpy.clock
import rclpy.time

class TestImageSave(unittest.TestCase):


    def set_layers(self, paths, data):
        layers = []

        for key in paths:
            layer = ImageLayer()
            im_cv2 = cv2.imread(paths[key], cv2.IMREAD_GRAYSCALE)
            layer.image = CvBridge().cv2_to_imgmsg(im_cv2, encoding="passthrough")
            layer.frame_id = str(data['frame_id'])
            layer.count = data['encoder_counts'][key]
            layer.mm = data['encoder_mm'][key]
            layer.name = key
            layers.append(layer)

        return layers

    def setUp(self) -> None:

        paths = {
            'top1' : '/home/wesley/traina/source_control/4_18_2024/threading_ws/src/image_service/test/old_style_image/10487561/10487561_top_1.bmp',
            'bot1' : '/home/wesley/traina/source_control/4_18_2024/threading_ws/src/image_service/test/old_style_image/10487561/10487561_bot_1.bmp',
            'bot2' : '/home/wesley/traina/source_control/4_18_2024/threading_ws/src/image_service/test/old_style_image/10487561/10487561_bot_2.bmp',
        }
        
        with open('/home/wesley/traina/source_control/4_18_2024/threading_ws/src/image_service/test/old_style_image/10487561/10487561.txt') as fp:
            data = json.load(fp)
            print(data)

        self.im_set = ImageSetHdr()

        self.im_set.frame_id = data['frame_id']
        self.im_set.stamp = rclpy.clock.Clock().now().to_msg()
        self.im_set.images = self.set_layers(paths, data)


        rclpy.init()
        self.node = ImageServices() #rclpy.create_node('image_services')

        return super().setUp()
    

    def test_image_saving(self):
       
       self.node.base_path = '/home/wesley/Pictures/'
       self.node.enable_saving = True
       self.node.save_set_callback(self.im_set)
       

       rclpy.spin_once(self.node)

       pass

if __name__ == '__main__':
    unittest.main()