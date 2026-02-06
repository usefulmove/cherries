import rclpy
from rclpy.node import Node

from rclpy.action import ActionClient

import rclpy.qos
from std_msgs.msg import String
from std_msgs.msg import Bool
from sensor_msgs.msg import Image

# some
from cherry_interfaces.action import FindCherries
from cherry_interfaces.srv import Detection
from cherry_interfaces.msg import ImageSet
from cherry_interfaces.msg import CherryArrayStamped
from cherry_interfaces.msg import Cherry
from cherry_interfaces.msg import EncoderCount
from cherry_interfaces.srv import GetCherryBuffer
from cherry_interfaces.srv import LatchRobot

import socket
import sys
import threading
import socket
import math

from cv_bridge import CvBridge
import cv2
import numpy as np
import time

# these constants
# kinda sorta, there are no constants in python so the capitals are a hint
# these shouldn't change...  please and thank you!
TRIGGER_MESSAGE = 'RUNFIND\r'.encode('utf-8')
OK_MESSAGE = 'OK\r'.encode('utf-8') # 'OK,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\r'.encode('utf-8')
ERROR_MESSAGE = 'ER\r'.encode('utf-8')
# images taken about every 150 mm, so if this distance is 
# exceeded by 3xm, then these is an issue
WATCHDOG_LENGTH = 3 * 150 

class FanucComms(Node):



    def __init__(self):
        self.count = 0
        self.br = CvBridge()

        super().__init__('fanuc_comms')
        self.publisher_ = self.create_publisher(String, 'tcp_incoming', 10)

        # create a client to handle getting images
        self._action_client = ActionClient(self, FindCherries, 'find_cherries')

        # monitor encoder to check and see if we are receiving reasonably spaced
        # triggers
        plc_io_qos = rclpy.qos.qos_profile_sensor_data
        plc_io_qos.lifespan = rclpy.qos.Duration(seconds=1)
        plc_io_qos.deadline = rclpy.qos.Duration(seconds=1)

        self.fanuc_alive = True
        self.current_mm = 0
        self.last_mm = math.pow(2, 31) # make this large to start, will get 
        # adjusted on first read of encoder value.

        self.last_type = 4
        self.mode = 'stems'

        self.encoder_subscriber = self.create_subscription(
            EncoderCount, 
            'encoder', 
            self.encoder_callback,
            plc_io_qos
        )

        self.buffer_client = self.create_client(
            GetCherryBuffer,
            'cherry_buffer/get',
        )

        self.latch_client = self.create_client(
            LatchRobot,
            'latch_robot',
        )

        self.robot_liveliness_publisher = self.create_publisher(Bool, 'fanuc_robot/alive', 10)
        timer_period = 1  # seconds
        self.liveliness_timer = self.create_timer(timer_period, self.liveliness_timer_callback)

        # start a thread that accepts connections.
        t = threading.Thread(target=self.connection_handler)
        t.start()

    def encoder_callback(self, encoder_message):
        self.current_mm = encoder_message.mm

        if (self.last_mm > self.current_mm):
            # we are goin backwards?
            # or perhaps the encoder value rolled over
            # make sure last is always smaller than current
            self.last_mm = self.current_mm

        if ((self.current_mm - self.last_mm) > WATCHDOG_LENGTH):
            self.fanuc_alive = False
        else:
            self.fanuc_alive = True

    def liveliness_timer_callback(self):
        msg = Bool()
        msg.data = self.fanuc_alive
        self.robot_liveliness_publisher.publish(msg)


    def connection_handler(self):
        # Create a TCP/IP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Bind the socket to the port
        server_address = ('10.0.0.10', 59002)
        self.get_logger().info('starting up on %s port %s' % server_address)
        sock.bind(server_address)

        # Listen for incoming connections
        sock.listen(1)

        # accept connections and get data
        # if the connection closes or errors, start listenting for a new connection.
        while True:
            # Wait for a connection
            self.get_logger().info('waiting for a connection')
            self.connection, client_address = sock.accept()
            try:
                self.get_logger().info('connection from {}'.format( client_address))
                # Receive the data in small chunks and retransmit it
                while True:
                    data = self.connection.recv(255)
                    self.get_logger().info('received "{}"'.format( data))
                    #print >>sys.stderr, 'received "%s"' % data
                    if data:
                        #print >>sys.stderr, 'sending data back to the client'
                        self.data_handler(data)

                    else:
                        # print >>sys.stderr, 'no more data from', client_address
                        self.get_logger().info('no more data from {}'.format(client_address))
                        break
            except Exception as e:
                self.get_logger().error('connection error: {}'.format(e))
            finally:
                # Clean up the connection
                self.get_logger().info('closed connection from {}'.format(client_address))
                self.connection.close()
            #self.get_logger().info('testing')

    def data_handler(self, data):
        try:

        # publish what we got

        # put the data in string message.
        # ROS2 wants a message type when publishing stuff.
        # i though you could do a primitive, but maybe not?  or is string not a primitive?
        # anyways, this works so...
            msg = String()
            msg.data = data.decode()

            # mayeb look at just beginning #TRIGGER_MESSAGE == data[0:7]
            # or maybe just trigger
            if TRIGGER_MESSAGE == data:

                # let system know the caemra has triggered
                # msg = Bool()
                # msg.data = True
                # self.encoder_latch_publisher_.publish(msg)

                self.id = 0

                # check for ID number, use an internal count if none specified
                try:
                    if len(data)>8:
                        data_string = data.decode()
                        split_data = data_string.split(',')
                        self.id = int(split_data[1])
                    else:
                        self.count = self.count + 1
                        self.id = self.count



                    # send an OK message. triggering was ok
                    # now we wait for acquisition and processing.
                    # should I wait till the camera and detection returns? maybe?
                    self.get_logger().info('repsonding with "{}"'.format(OK_MESSAGE))
                    self.connection.sendall(OK_MESSAGE)

                    # send a singal to the robot to synchronize things
                    latch_result = self.latch_client.call(LatchRobot.Request())

                    # pull data from the buffer
                    buffer_request = GetCherryBuffer.Request()
                    buffer_request.reference_mm = latch_result.encoder_mm
                    self.get_logger().info('latched robot at: {}'.format(latch_result.encoder_mm))
                    buffer_response = self.buffer_client.call(buffer_request)
                    self.get_logger().info('cherry array: {}'.format(buffer_response.cherry_array.cherries))
                    
                    # send the data to the robot
                    self.send_tcpip_messages(buffer_response.cherry_array.cherries, buffer_response.model_id)

                    # send some dummy data
                    #self.dummy_data()

                    #FanucComms.format_data()
                except Exception as e:
                    self.get_logger().error('could not parse data for id number. data: {}, error: {}'.format(data, e))
                    self.connection.sendall(ERROR_MESSAGE)

            else:
                self.get_logger().info('repsonding with "{}"'.format(ERROR_MESSAGE))
                self.connection.sendall(ERROR_MESSAGE)


        except Exception as e:
            self.get_logger().error('error occured. could not process data: {}, error: {}'.format(data, e))

    def dummy_data(self):
        # string_messages = []

        # cherry_count = 7
        # if cherry_count < 0:
        #     cherry_count = 0

        # this_msg = "{},{},".format(1,7)
        # for j in range (0,7):
        #     arr_index = j
        #     if arr_index >= len(arr_index):
        #         this_msg.append("0,0,")
        #     else:
        #         this_msg.append("{},", arr[arr_index])
        # if arr_index >= len(arr_index):
        #     this_msg.append("0,0\r")
        # else:
        #     this_msg.append("{}\r", arr[arr_index])

        # string_messages.append(this_msg)

        # # send the messasges
        # for rmsg in string_messages:

        time.sleep(0.25)


        rmsg = '{},2,3,4,5,6,7,8,9,10,11,12,13,14\r'.format(self.id)
        encoded_message = rmsg.encode('utf-8')
        self.connection.sendall(encoded_message)
        self.get_logger().info('sent data {} end value: {}'.format(encoded_message, encoded_message[-1]))

        # turn off the latch but
        msg = Bool()
        msg.data = False
        self.encoder_latch_publisher_.publish(msg)



    def trigger(self, id):

        # note the last trigger location, this is used to determine if 
        # the robot is alive
        self.last_mm = self.current_mm 

        try:
            self.get_logger().info('frame {} -> Fanuc trigger '.format(id))
            self._action_client.wait_for_server()

            # create goal
            goal_msg = FindCherries.Goal()
            goal_msg.frame_id = id

            # send goal
            self._send_goal_future = self._action_client.send_goal_async(goal_msg)
            self._send_goal_future.add_done_callback(self.goal_response_callback)
        except Exception as e:
            self.get_logger().error('frame {} -> trigger error: {}'.format(id, e))
            self.connection.sendall(ERROR_MESSAGE)
            self.get_logger().info('sent err result to fanuc {}'.format(ERROR_MESSAGE))
        return

    def goal_response_callback(self, future):
        try:
            #print('testing 2')
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().info('Frame trigger rejected !!! {}'.format(goal_handle))
                self.connection.sendall(ERROR_MESSAGE)
                return

            #print('Frame trigger accepted !!! {}'.format(goal_handle))

            #self.get_logger().info('Goal accepted')

            self._get_result_future = goal_handle.get_result_async()
            self._get_result_future.add_done_callback(self.get_result_callback)
        except Exception as e:
            self.get_logger().error('Error in goal_response_callback! error: {}'.format(e))
            self.connection.sendall(ERROR_MESSAGE)
            self.get_logger().info('sent err result to fanuc {}'.format(ERROR_MESSAGE))

    def get_result_callback(self, future):
        #print('testing 3')
        try:
            msg = future.result()

            status = msg.status
            result = msg.result  # this looks like a image_set
            self.get_logger().info('msg: {}'.format(result.cherries))
            #print(len(result.image_top.data))
            #print(len(result.image_back.data))


            self.get_logger().info('frame {} -> Aquired Image'.format(result.frame_id))
            #self.get_logger().info('image one size : {}'.format(result.image_top.height, result.image_top.width))
            #self.get_logger().info('image two size: {}'.format(result.image_back.height, result.image_top.width))
        except:
            self.get_logger().error('frame ?? -> Error processing image acquisition message!')

            self.connection.sendall(ERROR_MESSAGE)
            self.get_logger().info('sent err result to fanuc {}'.format(ERROR_MESSAGE))
            return

        # try:
        #     self.get_logger().info('repsonding with "{}"'.format(OK_MESSAGE))
        #     self.connection.sendall(OK_MESSAGE)
        # except Exception as e:
        #     self.get_logger().error('frame {} -> error sending OK_MESSAGE: {} '.format(id, e))

        #     self.connection.sendall(ERROR_MESSAGE)
        #     self.get_logger().info('sent err result to fanuc {}'.format(ERROR_MESSAGE))
        #     return
    
        try:
            #try:
            self.send_tcpip_messages(result.cherries) #, self.get_logger())
            # self.send_tcpip_messages(msg.cherries) #, self.get_logger())
            #   #self.get_logger().info('frame {} -> cherries: {}'.format(result.header.frame_id, frame.cherry_list))
            # except Exception as e:
            #     self.get_logger().error('frame {} -> failed to send message to fanuc robot. error: {}'.format(frame_id, e))

            # TO DO: publish the cherry results so that the control node can use them to draw sutff
        except Exception as e:
            self.get_logger().error('frame {} -> detection callback error: {} '.format(id, e))

            self.connection.sendall(ERROR_MESSAGE)
            self.get_logger().info('sent err result to fanuc {}'.format(ERROR_MESSAGE))
        return



    #def set_cherries(self, cherries, logger):
    def send_tcpip_messages(self, cherries, model_id=1):
        #print('set function: ', msg.cherries)
        # create stings with 'x,y' as integers. should be in mm, we don;t realy need more accuracy
        # and this approach should require less characters.  we are limit to 255 per message.
        # self.get_logger().info('frame {} -> cherry: {}'.format(self.id, cherries))

        # 1 is split the data between the robots
        # 3 is send data only to robot 1
        # 4 is send data only to robot 2
        arr = []
        for cherry in cherries:
            #self.get_logger().info('frame {} -> cherry: {}'.format(self.id, cherry))
            type_i = int.from_bytes(cherry.type, "big")
            # they should not be type 1 - aka clean cherries

            # note: need to filter in the buffer now
            #if (type_i != 1 and type_i != 3 and type_i != 5):
            x = int(cherry.x * 1000) - 155
            y = int(cherry.y * 1000) - 457

            xy_string = '{},{}'.format(x,y)
            
            # type_f = float(type_i)
            self.get_logger().info('frame {} -> cherry: {}'.format(self.id, xy_string))
            arr.append(xy_string)


        # create msg strings with 7 xy points per message.  start each message with number of valid points
        # string_messages = []

        #self.get_logger().info('frame {} -> arr data: {}'.format(self.id, arr))


        #print('frames_ceil: ', frames)

        # first message has the model_id and

        # # ********  STEMS_TEST ****************

        # # codes
        # #   1: split between robots
        # #   2: use robot 1
        # #   3: use robot 2

        # # flip flop on robot 1 vs robot 2
        # if self.last_type == 4:
        #     arr = ["0,-100","0,-200"]
        #     # dummy count data to make sure this is working as expected
        #     self.last_type = 3

        # else:
        #     self.last_type = 4
        #     arr = ["0,100","0,200"]

        # # self.last_type = 1

        # # ********  STEMS_TEST ****************

        # send up to 48 cherries
        cherry_count = len(arr)
        #frames_float = (cherry_count - 6) / 7 + 1
        #print('frames_float: ', frames_float)
        frames = math.ceil( (cherry_count - 6) / 7 + 1)

        rmsg = "{mode},{count},".format(mode=model_id, count=cherry_count) # .format(self.id, cherry_count, ...
        frame_start = 0
        frame_end = 5
        for j in range (frame_start,frame_end):
            arr_index = j
            if arr_index >= cherry_count:
                rmsg = rmsg + "0,0,"
            else:
                rmsg = rmsg + "{},".format(arr[arr_index])
        if (frame_end) < cherry_count:
            rmsg = rmsg + "{}\r".format( arr[frame_end])
        else:
            rmsg = rmsg + "0,0\r"  
        self.send_string(rmsg)

        # 2+ messages have up to 7 xy
        for k in range(0,frames-1):
            rmsg = ""
            frame_start = k * 7 + 6
            frame_end = frame_start + 6
            for j in range (frame_start,frame_end):
                arr_index = j
                #print(j, cherry_count)
                if arr_index >= (cherry_count ):
                    rmsg = rmsg + "0,0,"
                else:
                    rmsg = rmsg + "{},".format(arr[arr_index])
            if frame_end < cherry_count:
                rmsg = rmsg + "{}\r".format( arr[frame_end])
            else:
                rmsg = rmsg + "0,0\r"

            self.send_string(rmsg)

    def send_string(self, rmsg):
        #rmsg = 'id,count,x1,x2...,x7,y7\r'.format(self.id)
        encoded_message = rmsg.encode('utf-8')
        self.get_logger().info('encoded data {}'.format(encoded_message))
        try: 
            self.connection.sendall(encoded_message)
        except Exception as e:
            self.get_logger().error('Error sending data: {}'.format(e))




def main(args=None):
    rclpy.init(args=args)

    fanuc_comms = FanucComms()

    rclpy.spin(fanuc_comms)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    fanuc_comms.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
