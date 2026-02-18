# this recevies a color image with the red channel as the top image
# and the bllue channel as the bottom image

from cherry_interfaces.srv import Detection
from cherry_interfaces.msg import Cherry, CherryArray, CherryArrayStamped

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

# from .blob_detector import blob_detector
from sensor_msgs.msg import Image
import torch

import os

# abspath = os.path.abspath(__file__)
# dname = os.path.dirname(abspath)
# os.chdir(dname)
import sys
# caution: path[0] is reserved for script path (or '' in REPL)


# from cherry_detection.ai_detector import ai_detector_class

from .ai_detector import ai_detector_class
# import ai_detector
# from ai_detector import ai_detector_class

from ament_index_python.packages import get_package_share_directory


class Detector:  #
    def __init__(self):
        # self.detector = blob_detector((2448,652,math.pi/2), 2646.54418197725, ((27,188),(2448,611)))
        # blob_detector((2448,652,math.pi/2), 2710.31633094056, ((27,188),(2448,611)))

        # self.detector = ai_object_detector.yolov7_detector((2448,652,math.pi/2), 2710.31633094056, ((27,188),(2448,611)))

        # self.roi = roi
        self.origin = (2448, 652, math.pi / 2)
        self.scaling_factor = 2710.31633094056
        self.roi = ((27, 188), (2448, 611))

        # define rotation for converting to real world units
        rotation_roll = np.array(
            [
                [1, 0, 0],
                [0, math.cos(math.pi), -math.sin(math.pi)],
                [0, math.sin(math.pi), math.cos(math.pi)],
            ]
        )
        # print('roll rotation', rotation_roll)

        rotation_pitch = np.array(
            [
                [math.cos(self.origin[2]), -math.sin(self.origin[2]), 0],
                [math.sin(self.origin[2]), math.cos(self.origin[2]), 0],
                [0, 0, 1],
            ]
        )

        self.rotation = rotation_roll.dot(rotation_pitch)

        # get the weights from the package share directory
        # package_share_directory = '/home/user/cherry_ws/install/cherry_detection/share/cherry_detection/
        # get the weights from the package share directory
        package_share_directory = get_package_share_directory("control_node")

        # set up the the segmentation
        weight_path = os.path.join(package_share_directory, "cherry_segmentation.pt")
        weights = torch.load(weight_path)

        # set up the classifier
        weight_path2 = os.path.join(package_share_directory, "cherry_classification.pt")
        weights2 = torch.load(weight_path2)

        self.detector = ai_detector_class(weights, weights2)

        # self.publisher_kp_image_color_ = self.create_publisher(Image, '~/keypoint_image_color', 10 )
        # self.publisher_kp_image_processed_ = self.create_publisher(Image, '~/keypoint_image_processed', 10 )

    def scale(self, point):
        return (point[0] / self.scaling_factor, point[1] / self.scaling_factor)

    def rotate(self, point):
        # translate point and add Z element (always 0)
        point_px = np.array([point[0] - self.origin[0], point[1] - self.origin[1], 0])

        # perform roation using matrixes above

        # print('all rotation\n', rotation)
        # print('pixel point\n', point_px)
        point_world = self.rotation.dot(point_px)

        # print('rotated point\n', point_world)

        # only return the xy component and ignore the z
        return (point_world[0], point_world[1])

    def process_point(self, pt):
        pt_rotated = self.rotate(pt)
        pt_scaled = self.scale(pt_rotated)  # get in meters

        return pt_scaled

    def process_bbox(self, bbox):
        xy1 = self.process_point((bbox[0], bbox[1]))
        xy2 = self.process_point((bbox[2], bbox[3]))

        return [xy1[0], xy1[1], xy2[0], xy2[1]]

    # x, y, offset in pixels to origin
    # rotaiton in radains
    # scaling factor in pixel/ mm
    def real_world(self, detections):
        # we have a rotation around Z, the origin_rotation
        # and a rotation around x by 180 deg - the direction fo the y axis is flipped

        px_boxes = detections["boxes"]

        real_boxes = []
        for bbox in px_boxes:
            scaled = self.process_bbox(bbox)
            real_boxes.append(scaled)

        detections["real_boxes"] = real_boxes

        return detections

    def detect(self, image_color, logger):
        kp_im_procssed = None
        cherries = []
        # try:

        # process th e image
        dets, kp_im_procssed = self.detector.detect(image_color)
        if logger is not None:
            logger.info("detector -> processed image")

        dets = self.real_world(dets)
        if logger is not None:
            logger.info("detector -> created real world")
        # set the response message

        for i in range(len(dets["labels"])):
            xyxy = dets["real_boxes"][i]  # scaled is index 3
            confidence = dets["confidences"][i]
            cls = int(dets["labels"][i])
            print()
            # if confidence > 0.5 and cls == 1:
            # print()
            cherry = Cherry()
            cherry.x = (xyxy[0] + xyxy[2]) / 2.0
            cherry.y = (xyxy[1] + xyxy[3]) / 2.0
            cherry.type = (cls).to_bytes(1, "big")  # 1 is ok, 2 has a pit
            cherries.append(cherry)
            print(cherry)
        if logger is not None:
            logger.info("detector -> created rcherry list")
        # img_msg_color = self.br.cv2_to_imgmsg(kp_im_color, encoding='bgr8')
        # self.publisher_kp_image_color_.publish(img_msg_color)

        # except Exception as e:
        #     pass

        return cherries, kp_im_procssed


def main():
    my_detector = Detector()

    img = cv2.imread(
        "/home/user/Pictures/lots/cherries/20221102/image_20221102T161316.png"
    )
    # img = cv2.imread('/home/user/Pictures/cherries/image_20221027T125232.png')
    # img = cv2.imread('/home/user/Pictures/cherries/small.png')

    # img = cv2.imread('/media/user/FADC0612DC05CA39/traina/instance_segmentation/task_segment-2022_10_19_18_25_33-coco 1.0/images/20220919T093001_color.jpg')
    # img = cv2.imread('/media/user/FADC0612DC05CA39/traina/instance_segmentation/task_segment-2022_10_19_18_25_33-coco 1.0/images/20220919T100642_color.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    detections, img_labeled = my_detector.detect(img, None)

    for det in detections:
        x = det.x
        y = det.y
        c = int.from_bytes(det.type, "big")

        print(x, y, c)


if __name__ == "__main__":
    main()
