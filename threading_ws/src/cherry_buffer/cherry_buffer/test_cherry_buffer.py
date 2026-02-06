import rclpy
import unittest
from cherry_buffer import CherryBuffer
from sensor_msgs.msg import Image
from builtin_interfaces.msg import Time
from cherry_interfaces.msg import CherryArray, Cherry
from cherry_interfaces.srv import GetCherryBuffer
import os


class CherryBufferTestCase(unittest.TestCase):
    def setUp(self):
        self.cherry_buffer = CherryBuffer()
        self.cherry_array = CherryArray()
        self.cherry_array.encoder_count = 1000
        self.cherry_array.cherries = [
            Cherry(x=-307.0, y=-100.0, type=CherryBuffer.CHERRY_CLEAN),
            Cherry(x=-208.0, y=0.0, type=CherryBuffer.CHERRY_PIT),
            Cherry(x=-109.0, y=100.0, type=CherryBuffer.CHERRY_SIDE),
            Cherry(x=5.0, y=-100.0, type=CherryBuffer.CHERRY_MAYBE),
            Cherry(x=6.0, y=0.0, type=CherryBuffer.CHERRY_STEM),
            Cherry(x=7.0, y=100.0, type=CherryBuffer.CHERRY_CLEAN),
            Cherry(x=301.0, y=-100.0, type=CherryBuffer.CHERRY_PIT),
            Cherry(x=203.0, y=0.0, type=CherryBuffer.CHERRY_MAYBE),
            Cherry(x=105.0, y=100.0, type=CherryBuffer.CHERRY_CLEAN),
        ]
        # makea  second instance, since the msg gets manipulated in add_to_buffer
        # not a huge deal maybe..   
        self.cherry_array2 = CherryArray()
        self.cherry_array2.encoder_count = 1000
        self.cherry_array2.cherries = [
            Cherry(x=-307.0, y=-100.0, type=CherryBuffer.CHERRY_CLEAN),
            Cherry(x=-208.0, y=0.0, type=CherryBuffer.CHERRY_PIT),
            Cherry(x=-109.0, y=100.0, type=CherryBuffer.CHERRY_SIDE),
            Cherry(x=5.0, y=-100.0, type=CherryBuffer.CHERRY_MAYBE),
            Cherry(x=6.0, y=0.0, type=CherryBuffer.CHERRY_STEM),
            Cherry(x=7.0, y=100.0, type=CherryBuffer.CHERRY_CLEAN),
            Cherry(x=301.0, y=-100.0, type=CherryBuffer.CHERRY_PIT),
            Cherry(x=203.0, y=0.0, type=CherryBuffer.CHERRY_MAYBE),
            Cherry(x=105.0, y=100.0, type=CherryBuffer.CHERRY_CLEAN),
        ]

    def test_add_to_buffer(self):
        self.cherry_buffer.add_to_buffer(self.cherry_array)
        popped = self.cherry_buffer.pop_from_buffer(1000)
        self.assertEqual(self.cherry_array, popped)

        self.cherry_buffer.add_to_buffer(self.cherry_array)
        popped = self.cherry_buffer.pop_from_buffer(1200)
        for argi in range (0, len(self.cherry_array.cherries)):
            self.assertEqual(
                self.cherry_array2.cherries[argi].x + 200,
                popped.cherries[argi].x
            )
            self.assertGreater(
                popped.cherries[argi].x,
                self.cherry_array2.cherries[argi].x
            )

    def test_services(self):

        msg = CherryArray()
        msg.encoder_count = 1000
        msg.cherries = [
            Cherry(x= 1.0, y= 2.0, type=CherryBuffer.CHERRY_CLEAN),
            Cherry(x=  3.0, y=    4.0, type=CherryBuffer.CHERRY_PIT),
            Cherry(x=  -100.0, y=    5.0, type=CherryBuffer.CHERRY_MAYBE),
        ]
        #print(msg.cherries[0].x, msg.cherries[0].y, msg.cherries[0].type)
        #print(msg.cherries[1].x, msg.cherries[1].y, msg.cherries[1].type)

        self.cherry_buffer.detection_callback(msg)
        response = GetCherryBuffer.Response()
        self.cherry_buffer.get_callback(
            GetCherryBuffer.Request(reference_mm = 1300),
            response
        )
        #print(response.cherry_array.cherries[0].x, response.cherry_array.cherries[0].y, response.cherry_array.cherries[0].type)
        #print(response.cherry_array.cherries[1].x, response.cherry_array.cherries[1].y, response.cherry_array.cherries[1].type)
        self.assertEqual(response.cherry_array.cherries[0].x, 301)
        self.assertEqual(response.cherry_array.cherries[1].x, 303)
        self.assertEqual(response.cherry_array.cherries[2].x, 200)


    def __del__(self):
        try:
            self.cherry_buffer.destroy_node()
            
        except:
            pass


def main():
    rclpy.init()
    unittest.main()
    rclpy.shutdown()

if __name__ == '__main__':
    main()