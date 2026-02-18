# this recevies a color image with the red channel as the top image
# and the bllue channel as the bottom image

from cherry_interfaces.srv import Detectionhdr
from cherry_interfaces.msg import Cherry, CherryArray, ImageSetHdr, ImageLayer, Trigger
from rcl_interfaces.msg import SetParametersResult

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


from cherry_detection.ai_detector import ai_detector_class
from cherry_detection.ai_detector2 import ai_detector_class_2
from cherry_detection.ai_detector3 import ai_detector_class_3
from cherry_detection.ai_detector4 import ai_detector_class_4

from ament_index_python.packages import get_package_share_directory


algorithms = [
    "fasterRCNN-Mask_ResNet50_V1",
    "fasterRCNN-NoMask_ResNet50_6-12-2023",
    "newlight-mask-12-15-2023",
    "newlights-nomask-12-15-2023",
    "NMS-nomask-1-3-2024",
    "hdr_v1",
    "hdr_v2",
    "vote_v1",
]


class DetectorNode(Node):
    def __init__(self):
        super().__init__("detection_server")
        self.srv = self.create_service(
            Detectionhdr, "~/detect", self.detect_with_blob_callback
        )
        self.br = CvBridge()
        # self.detector = blob_detector((2448,652,math.pi/2), 2646.54418197725, ((27,188),(2448,611)))
        # blob_detector((2448,652,math.pi/2), 2710.31633094056, ((27,188),(2448,611)))

        # self.detector = ai_object_detector.yolov7_detector((2448,652,math.pi/2), 2710.31633094056, ((27,188),(2448,611)))

        # self.roi = roi
        self.origin = (2448, 652, math.pi / 2)
        self.scaling_factor = 2710.31633094056
        self.roi = ((27, 188), (2448, 611))

        self.simulate_detection = False
        self.simulate_count = 0
        self.algorithm = "nothing"
        self.declare_parameter("simulate_detection", self.simulate_detection)
        self.declare_parameter("algorithm", self.algorithm)

        self.declare_parameter("maybe_threshold", 0.04)
        self.declare_parameter("pick_threshold", 0.06)
        self.add_on_set_parameters_callback(self.parameters_callback)

        self.load_v6()

        # update the parameter after loading v6
        param_update = rclpy.Parameter(
            "algorithm", rclpy.Parameter.Type.STRING, self.algorithm
        )
        self.set_parameters([param_update])

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

        self.publisher_kp_image_color_ = self.create_publisher(
            Image, "~/keypoint_image_color", 10
        )
        self.publisher_kp_image_processed_ = self.create_publisher(
            Image, "~/keypoint_image_processed", 10
        )
        self.publisher_detections_ = self.create_publisher(
            CherryArray, "~/detections", 10
        )

        self.subcription_image_set_ = self.create_subscription(
            ImageSetHdr, "image_set", self.image_set_callback, 10
        )

    def parameters_callback(self, params):
        for param in params:
            if param.name == "simulate_detection":
                try:
                    self.get_logger().info(
                        "Set 'simulate_encoder' to {}".format(str(param.value))
                    )
                    self.simulate_detection = param.value
                    # self.simulate_ticks()  # start the simulated upadte sequence

                except:
                    self.get_logger().error(
                        "Could not set 'simulate_encoder' to {}".format(param.value)
                    )
                    return SetParametersResult(successful=False)

            if param.name == "algorithm":
                try:
                    if param.value == "fasterRCNN-Mask_ResNet50_V1":
                        self.load_v1()
                    elif param.value == "fasterRCNN-NoMask_ResNet50_6-12-2023":
                        self.load_v2()
                    elif param.value == "newlight-mask-12-15-2023":
                        self.load_v3()
                    elif param.value == "newlights-nomask-12-15-2023":
                        self.load_v4()
                    elif param.value == "NMS-nomask-1-3-2024":
                        self.load_v5()
                    elif param.value == "hdr_v1":
                        self.load_v6()
                    elif param.value == "hdr_v2":
                        self.load_v7()
                    elif param.value == "vote_v1":
                        self.load_v8()

                    else:
                        self.get_logger().error(
                            "Valid algorithm names: {}".format(algorithms)
                        )
                        return SetParametersResult(successful=False)

                except Exception as e:
                    self.get_logger().error(
                        "Could not set 'algorithm' to {}".format(param.value)
                    )
                    self.get_logger().error("Exception: {}".format(e))
                    return SetParametersResult(successful=False)
            elif param.name == "maybe_threshold":
                try:
                    if self.detector.pick_threshold < param.value:
                        self.get_logger().info(
                            "cannot set maybe value greater than pick_theshold".format(
                                str(param.value)
                            )
                        )
                        return SetParametersResult(successful=False)

                    self.get_logger().info(
                        "Set 'maybe_threshold' to {}".format(str(param.value))
                    )
                    self.detector.maybe_threshold = param.value
                except:
                    self.get_logger().error(
                        "Could not set 'maybe_threshold' to {}".format(param.value)
                    )
                    return SetParametersResult(successful=False)
            elif param.name == "pick_threshold":
                try:
                    if self.detector.maybe_threshold > param.value:
                        self.get_logger().info(
                            "cannot set maybe value less than maybe_theshold".format(
                                str(param.value)
                            )
                        )
                        return SetParametersResult(successful=False)

                    self.get_logger().info(
                        "Set 'pick_threshold' to {}".format(str(param.value))
                    )
                    self.detector.pick_threshold = param.value

                except:
                    self.get_logger().error(
                        "Could not set 'pick_threshold' to {}".format(param.value)
                    )
                    return SetParametersResult(successful=False)
            else:
                self.get_logger().error("Unkown paramter: {}".format(param.name))
                return SetParametersResult(successful=False)

        return SetParametersResult(successful=True)

    def load_v1(self):
        # get the weights from the package share directory
        package_share_directory = get_package_share_directory("cherry_detection")
        weight_path = os.path.join(package_share_directory, "cherry_segmentation.pt")
        # weight_path = 'segmentation_20.pt'
        weights = torch.load(weight_path)

        # set up the classifier
        weight_path2 = os.path.join(package_share_directory, "cherry_classification.pt")
        weights2 = torch.load(weight_path2)

        self.detector = ai_detector_class(weights, weights2)
        self.algorithm = "fasterRCNN-Mask_ResNet50_V1"

    def load_v2(self):
        package_share_directory = get_package_share_directory("cherry_detection")

        weight_path = os.path.join(
            package_share_directory, "seg_model_fasterRCNN_2_23_2022.pt"
        )
        weights = torch.load(weight_path)

        # set up the classifier
        weight_path2 = os.path.join(
            package_share_directory, "classification-6-11-2023.pt"
        )
        weights2 = torch.load(weight_path2)

        self.detector = ai_detector_class_2(weights, weights2)
        self.algorithm = "fasterRCNN-NoMask_ResNet50_6-12-2023"

    def load_v3(self):
        # get the weights from the package share directory
        package_share_directory = get_package_share_directory("cherry_detection")
        weight_path = os.path.join(package_share_directory, "cherry_segmentation.pt")
        # weight_path = 'segmentation_20.pt'
        weights = torch.load(weight_path)

        # set up the classifier
        weight_path2 = os.path.join(
            package_share_directory, "newlight-mask-12-15-2023.pt"
        )
        weights2 = torch.load(weight_path2)

        self.detector = ai_detector_class(weights, weights2)
        self.algorithm = "newlight-mask-12-15-2023"

    def load_v4(self):
        package_share_directory = get_package_share_directory("cherry_detection")

        weight_path = os.path.join(
            package_share_directory, "seg_model_fasterRCNN_2_23_2022.pt"
        )
        weights = torch.load(weight_path)

        # set up the classifier
        weight_path2 = os.path.join(
            package_share_directory, "newlights-nomask-12-15-2023.pt"
        )
        weights2 = torch.load(weight_path2)

        self.detector = ai_detector_class_2(weights, weights2)
        self.algorithm = "newlights-nomask-12-15-2023"

    def load_v5(self):
        package_share_directory = get_package_share_directory("cherry_detection")

        weight_path = os.path.join(
            package_share_directory, "seg_model_fasterRCNN_2_23_2022.pt"
        )
        weights = torch.load(weight_path)

        # set up the classifier
        weight_path2 = os.path.join(package_share_directory, "NMS-nomask-1-3-2024.pt")
        weights2 = torch.load(weight_path2)

        self.detector = ai_detector_class_2(weights, weights2)
        self.algorithm = "NMS-nomask-1-3-2024"

    def load_v6(self):
        package_share_directory = get_package_share_directory("cherry_detection")

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        weight_path = os.path.join(package_share_directory, "seg_model_red_v1.pt")
        weights = torch.load(weight_path, map_location=device)

        weight_path_stem = os.path.join(
            package_share_directory, "stem_model_10_5_2024.pt"
        )
        weights_stem = torch.load(weight_path_stem, map_location=device)

        # set up the classifier
        weight_path2 = os.path.join(
            package_share_directory, "classification-2_26_2025-iter5.pt"
        )
        weights2 = torch.load(weight_path2, map_location=device)

        self.detector = ai_detector_class_3(weights, weights2, weights_stem)
        self.algorithm = "hdr_v1"

    def load_v7(self):
        package_share_directory = get_package_share_directory("cherry_detection")

        weight_path = os.path.join(package_share_directory, "seg_model_red_v1.pt")
        weights = torch.load(weight_path)

        # set up the classifier
        weight_path2 = os.path.join(
            package_share_directory, "classification-202406031958-resnet50-adam-2.pt"
        )
        weights2 = torch.load(weight_path2)

        self.detector = ai_detector_class_3(weights, weights2)
        self.algorithm = "hdr_v2"

    def load_v8(self):
        package_share_directory = get_package_share_directory("cherry_detection")

        weight_path = os.path.join(package_share_directory, "seg_model_red_v1.pt")
        weights = torch.load(weight_path)

        # set up the classifier
        weight_path_resnet50 = os.path.join(
            package_share_directory, "classification-202406032102-resnet50-adam-1.pt"
        )
        weights_resnet50 = torch.load(weight_path_resnet50, torch.device("cpu"))
        weight_path_mobilenet = os.path.join(
            package_share_directory, "classification-202406032102-mobilenet-adam-1.pt"
        )
        weights_mobilenet = torch.load(weight_path_mobilenet, torch.device("cpu"))
        weight_path_densenet = os.path.join(
            package_share_directory, "classification-202406032102-densenet-adam-1.pt"
        )
        weights_densenet = torch.load(weight_path_densenet, torch.device("cpu"))

        self.detector = ai_detector_class_4(
            weights, weights_resnet50, weights_mobilenet, weights_densenet
        )
        self.algorithm = "vote_v1"

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
        # print('detections*****', detections)

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
        dets, kp_im_procssed, stems = self.detector.detect(image_color)

        dets = self.real_world(dets)
        if logger is not None:
            logger.info("detector -> created real world")
        # set the response message

        for i in range(len(dets["labels"])):
            xyxy = dets["real_boxes"][i]  # scaled is index 3
            confidence = dets["confidences"][i]
            cls = int(dets["labels"][i])
            # if confidence > 0.5 and cls == 1:
            # print()
            cherry = Cherry()
            cherry.x = (xyxy[0] + xyxy[2]) / 2.0
            cherry.y = (xyxy[1] + xyxy[3]) / 2.0
            cherry.type = (cls).to_bytes(1, "big")  # 1 is ok, 2 has a pit
            cherries.append(cherry)
            # print(cherry)

        if logger is not None:
            logger.info("detector -> created rcherry list")
        # img_msg_color = self.br.cv2_to_imgmsg(kp_im_color, encoding='bgr8')
        # self.publisher_kp_image_color_.publish(img_msg_color)

        # except Exception as e:
        #     pass

        return cherries, kp_im_procssed

    def simulate(self):
        self.simulate_count = self.simulate_count + 1

        if self.simulate_count > 4:
            self.simulate_count = 0

        kp_im_procssed = np.zeros(500, 2454, 3)
        cherries = []
        # try:

        offset = self.simulate_count * 0.025
        for i in range(0, 10):
            cherry = Cherry()
            cherry.x = i * 0.175 / 10
            cherry.y = 0.2 + i * 0.6 / 10
            cherry.type = (2).to_bytes(1, "big")  # 1 is ok, 2 has a pit
            cherries.append(cherry)
            # print(cherry)
        # if self.logger is not None:
        #     self.logger.info('simulator -> created cherry list')
        # img_msg_color = self.br.cv2_to_imgmsg(kp_im_color, encoding='bgr8')
        # self.publisher_kp_image_color_.publish(img_msg_color)

        return cherries, kp_im_procssed

    def image_set_hdr_subcription_callback(self, image_set_msg):
        pass

    def detections_to_cherry_msg(self, dets, stems):
        cherries = []
        if len(dets["labels"]) > 0:
            dets = self.real_world(dets)

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

        if len(stems["labels"]) > 0:
            stems = self.real_world(stems)
            stem_cls = 6
            for i in range(len(stems["labels"])):
                xyxy = stems["real_boxes"][i]  # scaled is index 3
                cls = int(stems["labels"][i])
                # if confidence > 0.5 and cls == 1:
                # print()
                cherry = Cherry()
                cherry.x = (xyxy[0] + xyxy[2]) / 2.0
                cherry.y = (xyxy[1] + xyxy[3]) / 2.0
                cherry.type = (stem_cls).to_bytes(1, "big")  # 1 is ok, 2 has a pit
                cherries.append(cherry)
            # print(cherry)

        return cherries

    def detect(self, image_set_hdr):
        cherries = []
        if self.simulate_detection == False:
            # run the detection as normal
            dets, kp_im_procssed, stems = self.detector.detect(image_set_hdr)
            if len(dets["labels"]) > 0 or len(stems["labels"]) > 0:
                cherries = self.detections_to_cherry_msg(dets, stems)
        else:
            # get some simulated cherries
            dets, kp_im_procssed = self.simulate()
            cherries = dets

        # publish the processed image
        img_msg_processed = self.br.cv2_to_imgmsg(kp_im_procssed, encoding="bgr8")
        self.publisher_kp_image_processed_.publish(img_msg_processed)

        # return the cherries to the caller
        return cherries

    def request_to_image_set(self, request: Detectionhdr.Request):
        image_set = ImageSetHdr()

        # TODO: switch to Trigger header for frame_id, encoder location, etc.
        image_set.trigger = Trigger()
        image_set.trigger.frame_id = request.frame_id
        image_set.trigger.stamp = (
            self.get_clock().now().to_msg()
        )  # use the current time i guess
        image_set.trigger.encoder_count = request.count_bot1
        image_set.trigger.encoder_mm = request.mm_bot1

        image_set.images = [ImageLayer(), ImageLayer(), ImageLayer()]

        image_set.images[0].frame_id = request.frame_id
        image_set.images[0].image = request.image_bot1
        image_set.images[0].name = "bot1"
        image_set.images[0].mm = request.mm_bot1
        image_set.images[0].count = request.count_bot1

        image_set.images[1].frame_id = request.frame_id
        image_set.images[1].image = request.image_bot2
        image_set.images[1].name = "bot2"
        image_set.images[1].mm = request.mm_bot2
        image_set.images[1].count = request.count_bot2

        image_set.images[2].frame_id = request.frame_id
        image_set.images[2].image = request.image_top2
        image_set.images[2].name = "top2"
        image_set.images[2].mm = request.mm_top2
        image_set.images[2].count = request.count_top2

        return image_set

    def publish_detections(self, cherries, encoder_count):
        try:
            # publish the results
            cmsg = CherryArray()
            cmsg.encoder_count = encoder_count
            cmsg.cherries = cherries
            self.publisher_detections_.publish(cmsg)
        except Exception as e:
            self.get_logger().error("publish detection! error: {}".format(e))

    def image_set_callback(self, image_set_hdr_msg: ImageSetHdr):
        im_dict = {}
        for im in image_set_hdr_msg.images:
            im_dict[im.name] = im

        self.get_logger().info(
            "detect frame {} with encoder mm: {}".format(
                image_set_hdr_msg.trigger.frame_id, image_set_hdr_msg.trigger.encoder_mm
            )
        )

        # image_set_hdr = self.request_to_image_set(image_set_hdr_msg)
        cherries = self.detect(image_set_hdr_msg)

        # needed for projectort to function
        # buffer also listens
        self.publish_detections(cherries, im_dict["bot1"].mm)

    def detect_with_blob_callback(self, request, response):
        # self.get_logger().info('start!' )
        # try:
        # get the color image
        response.encoder_count = request.mm_bot1
        self.get_logger().info(
            "detect frame with encoder count:".format(request.mm_bot1)
        )

        image_set_hdr = self.request_to_image_set(request)
        cherries = self.detect(image_set_hdr)
        response.cherries = cherries

        # needed for projectort yo function
        # self.publish_detections(cherries,request.mm_bot1)

        return response


def main():
    rclpy.init()

    minimal_service = DetectorNode()

    rclpy.spin(minimal_service)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
