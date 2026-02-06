
from email.mime import image
from genericpath import exists
from os import device_encoding
import os
from statistics import mode
import numpy as np
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
# from engine import train_one_epoch, evaluate
import torchvision.transforms as T
import cv2
from PIL import Image
from datetime import datetime
from cv_bridge import CvBridge
from cherry_interfaces.srv import Detectionhdr
from cherry_interfaces.msg import ImageSetHdr, ImageLayer

import random

import matplotlib.pyplot as plt
from ament_index_python.packages import get_package_share_directory
from torchvision.models import resnet50 #, ResNet50_Weights
import json

import time
#import functorch

import torchvision

# base_image = Image.open('/gdrive/MyDrive/traina/hdr/images/blurred.png')
# base_tensor = torchvision.transforms.ToTensor()(base_image)
# base_multipler = base_tensor.max() / base_tensor
# base_multipler = base_multipler[0, :, :] # only need one channel

# def correct_top(image):
#   im = image * base_multipler
#   im = im.clip(0,1)
#   return im
PIXEL_PER_COUNT = 0.24555903866248

def get_offset_image_test(im_path, offset_count, correct_top_flag=False):
  offset_pixel  = int(offset_count * PIXEL_PER_COUNT)

  im_pil = Image.open(im_path)
  im_tensor = torchvision.transforms.ToTensor()(im_pil)
  #print(im_tensor.shape)
#   if (correct_top_flag == True):
#     im_tensor = correct_top(im_tensor)
  #print(im_tensor.shape)
  # return image if no offset
  if offset_pixel == 0:
    return im_tensor

  # offset image otherwise
  im_offset = torch.zeros(im_tensor.shape)
  im_offset[:,offset_pixel:,:] = im_tensor[:,:-offset_pixel,:]

  # PIL.Image.fromarray(im_offset.squeeze().mul(255).byte().numpy())



  return im_offset

def get_test_msg(base_path, im_id):

    data_path = '{base}/{id}/{id}.txt'.format(base = base_path, id = im_id)
    #print( data_path)
    with  open(data_path) as f:
        data = json.load(f)

    paths = [
    '{base}/{id}/{id}_bot_1.bmp'.format(base = base_path, id = im_id),
    '{base}/{id}/{id}_bot_2.bmp'.format(base = base_path, id = im_id),
    '{base}/{id}/{id}_top_1.bmp'.format(base = base_path, id = im_id),
    '{base}/{id}/{id}_top_2.bmp'.format(base = base_path, id = im_id),
    ]
    im_tensors = torch.zeros(4, 500, 2464)

    im_tensors[0, :, :] = get_offset_image_test(paths[0], 0)
    im_tensors[1, :, :] = get_offset_image_test(paths[1], 0) #data['encoder_counts']['bot2'] - data['encoder_counts']['bot1'])
    im_tensors[2, :, :] = get_offset_image_test(paths[2], 0) #data['encoder_counts']['top1'] - data['encoder_counts']['bot1'], True)
    im_tensors[3, :, :] = get_offset_image_test(paths[3], 0) #data['encoder_counts']['top2'] - data['encoder_counts']['bot1'], True)

    im_np = im_tensors.numpy()

    msg = Detectionhdr.Request()

    bridge = CvBridge()


    msg.image_bot1 = bridge.cv2_to_imgmsg(im_np[0], encoding='passthrough')
    msg.image_bot2 = bridge.cv2_to_imgmsg(im_np[1], encoding='passthrough')
    msg.image_top1 = bridge.cv2_to_imgmsg(im_np[2], encoding='passthrough')
    msg.image_top2 = bridge.cv2_to_imgmsg(im_np[3], encoding='passthrough')

    msg.count_top1 = data['encoder_counts']['top1']
    msg.count_top2 = data['encoder_counts']['top2']
    msg.count_bot1 = data['encoder_counts']['bot1']
    msg.count_bot2 = data['encoder_counts']['bot2']

    msg.frame_id = msg.count_top1

    return msg

def plot_one_box(x, img, color=None, label=None, line_thickness=3):

    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def pil_to_tensor_gpu(pil_image, device):

    img_to_tensor = T.ToTensor()

    img_tensor  = img_to_tensor(pil_image)
    gpu_tensor = img_tensor.to(device)
    return gpu_tensor


def set_label(classification, val):
    # print(classification)
    classification['label'] = val

    return classification

# return True if cherry is in bounds
def check_location(box):
    # only look at things we are confident are cherries or pits

    box_int = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]

    # if too far up / down the conveyor, we will not process the cherry
    # if ( box_int[1] < 181 or box_int[3] > 706):
    #     location = 3
    # if too far on sides, we will not process the cherry, but flag as bad
    if ( box_int[0] < 174 or box_int[2] > 2253):
        return False

    #print(box, location)
    return True

class ai_detector_class_3:
    def __init__(self, weights, weights2, weights_stem):

        self.PIXEL_PER_COUNT = 0.24555903866248
        self.br = CvBridge()

        # get the device type
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print('cuda is avaliable', torch.cuda.is_available())

        # get the weights from the package share directory
        #package_share_directory = get_package_share_directory('cherry_detection')
        #weight_path = os.path.join(package_share_directory, 'segmentation_20.pt')
        #weight_path = 'segmentation_20.pt'

        #weights = torch.load(weight_path)
        self.pick_threshold = 0.06
        self.maybe_threshold = 0.04

        # get the model definition, load weights, and set to eval mode, move to gpu
        # we have two classes, 0: background, 1: cherry matter
        self.model = self.get_instance_segmentation_model(2)
        self.model.load_state_dict(weights)
        self.model.eval()
        self.model.to(self.device )
        
        self.stem_model = self.get_stem_model(weights_stem)
        self.stem_model.eval()
        self.stem_model.to(self.device)

        # define some transforms
        self.tesnor_to_img = T.PILToTensor() 



        self.classifier = resnet50().to(self.device)
        num_ftrs = self.classifier.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        self.classifier.fc = torch.nn.Linear(num_ftrs, 3)

        self.classifier.load_state_dict(weights2)
        self.classifier.eval()

        if torch.cuda.is_available():
            self.classifier.cuda()


        self.count_im = 0
        self.count_cherry = 0

    def get_stem_model(self, stem_weights):
        m = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=False)
        in_features = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        m.load_state_dict(stem_weights)
        return m

    # HxWxC 8bit [0 , 255] to CxHxW float [0., 1.]
    # nd_array or PIL image
    # to tensor converts HxWxC 8bit 0-225 to CxHxW float 0.-1.

    def get_instance_segmentation_model(self, num_classes):
        cherry_anchor_sizes = ((8,), (16,), (32,), (64,) )
        cherry_apect_ratios = ((0.666, 1.0, 1.333),) * len(cherry_anchor_sizes)
        cherry_rpn = AnchorGenerator(cherry_anchor_sizes, cherry_apect_ratios)

    # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(

            pretrained=False,
            num_classes=2,

            # base class MaskR-CNN parameters
            # transform parameters
            min_size=250,
            max_size=1232,
            image_mean=None,
            image_std=None,
            # RPN parameters
            rpn_anchor_generator=None,
            rpn_head=None,
            rpn_pre_nms_top_n_train=20000,
            rpn_pre_nms_top_n_test=10000,
            rpn_post_nms_top_n_train=20000,
            rpn_post_nms_top_n_test=10000,
            rpn_nms_thresh=0.7,
            rpn_fg_iou_thresh=0.7,
            rpn_bg_iou_thresh=0.3,
            rpn_batch_size_per_image=2560,
            rpn_positive_fraction=0.5,
            rpn_score_thresh=0.0,
            # Box parameters
            box_roi_pool=None,
            box_head=None,
            box_predictor=None,
            box_score_thresh=0.5,
            box_nms_thresh=0.5,
            box_detections_per_img=1000,
            box_fg_iou_thresh=0.5,
            box_bg_iou_thresh=0.5,
            box_batch_size_per_image=5120,
            box_positive_fraction=0.25,
            bbox_reg_weights=None,
            # Mask parameters
            mask_roi_pool=None,
            mask_head=None,
            mask_predictor=None,

        )

        return model        

    def tensor_to_cv2(self, img_tensor):

        # get transform and convert to PIL Image
        img_pil = self.tesnor_to_img(img_tensor)

        # convert to ndarray
        im_np = np.asarray(img_pil)

        return im_np


    def filter_prediction(self, prediction):
        masks = prediction['masks']
        scores = prediction['scores']
        boxes = prediction['boxes']


        # middle = []
        # sides = []
        # top_or_bot = []

        detection_list = []

        count = 0
        for index in range(0,len(masks)):

            mask = masks[index]

            #mask_image = topil(mask)
            mask_image = mask.cpu().numpy()
            mask_image  = mask_image.transpose((1, 2, 0))

            #mask_image = mask_image.convert('L')

            score = scores[index]




        
        return detection_list


    # used for debugging, this may need to be updated
    def save_classification_data(self, img, seg_score, label, label_prob, label_scores, label_probs, datetime_string):


        self.count_cherry = self.count_cherry + 1

        path_base  = f'/home/user/Pictures/cherry_classification/{datetime_string}'

        if not os.path.exists(path_base):
            # if does not exist create the folder
            os.makedirs(path_base, exist_ok=True)

        im_name = f'cherry_{self.count_cherry}.png'
        im_path = os.path.join(path_base, im_name)
        data_path = f'/home/user/Pictures/cherry_classification/{datetime_string}/data.json'
        cvs_path = f'/home/user/Pictures/cherry_classification/{datetime_string}/data.cvs'


        # save the image
        

        tf = T.transforms.Compose([
            T.ToPILImage(),

        ])

        #print('save', img.size())
        img_pil = tf(img)
        img_cv2 = np.array(img_pil) 
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)


        cv2.imwrite(im_path, img_cv2)


        locations = ['none', 'cherry_clean', 'cherry_pit', 'side', 'top/bot', 'maybe']

        # save the data
        data_dict = {
            'image' : im_name,
            'label' : str(label),
            'label_string' : locations[label],
            'seg_score' : str(seg_score),
            'label_prob' : str(label_prob),
            'label_scores' : str(label_scores),
            'label_probs' : str(label_probs),
        }

        #print(data_dict)
        
        with open(data_path, 'a') as data_file:
            line = json.dump(data_dict, data_file)
            #data_file.writeline(line)
            
        if not os.path.exists(cvs_path):
            with open(cvs_path, 'w') as data_file:
                #data_file.writeline(line)        
                cvsline = f'image_name,best_label_code,best_label_name,segmentation_score,best_label_probablity,clean_score,pit_score,clean_prob,pit_prob\n'
                data_file.write(cvsline)

        with open(cvs_path, 'a') as data_file:
            #data_file.writeline(line)        
            cvsline = f'{im_name},{label},{locations[label]},{seg_score},{label_prob},{label_scores[0]},{label_scores[1]},{label_probs[0]},{label_probs[1]}\n'
            data_file.write(cvsline)



    def get_sub_img(self, im_tensor, bbox):
        try:
            center = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
            bbox_100 = [int(center[0])-50, int(center[1])-50, int(center[0])+50, int(center[1])+50]
                
            bbox_tensor = im_tensor[:, int(bbox_100[1]):int(bbox_100[3]), int(bbox_100[0]):int(bbox_100[2])]

            bbox_tensor = bbox_tensor[ :, :100, :100] # make sure the sub image is 100x100

        except:
            bbox_tensor = torch.zeros(3, 100, 100) # if there is an isssue, make the a black image as a substitute.

        return bbox_tensor


        

    # expec numpy array of shape height, width, channels
    # 4 channel image weith channel 3 as the top image
    def get_red_image(self, im_cv):
        shape = im_cv.shape
        im_cv
        img = torch.zeros((shape[0], shape[1], 3))
        img[:, :, 0] = im_cv
        img[:, :, 1] = im_cv
        img[:, :, 2] = im_cv

        # retrun image with all 3 channels set to the color channl
        return img

    # 4 chanel image
    #  0: bot high
    #  1: bot low
    #  2: top high
    #  3: top low
    def get_color_image(self, im_cv):
        shape = im_cv.shape
        im_cv
        img = np.zeros((shape[0], shape[1], 3))
        img[:, :, 0] = im_cv[0]
        img[:, :, 1] = im_cv[1]
        img[:, :, 2] = im_cv[3]

        # retrun image with all 3 channels set to the color channl
        return img

    def get_offset_image(self, im_pil, offset_count, correct_top_flag=False):
        offset_pixel  = int(offset_count * self.PIXEL_PER_COUNT)

        im_tensor = torchvision.transforms.ToTensor()(im_pil)
        #print(im_tensor.shape)
        # if (correct_top_flag == True):
        #     im_tensor = correct_top(im_tensor)
        #print(im_tensor.shape)
        # return image if no offset
        if offset_pixel == 0:
            return im_tensor

        # offset image otherwise
        im_offset = torch.zeros(im_tensor.shape)
        im_offset[:,offset_pixel:,:] = im_tensor[:,:-offset_pixel,:]

        # PIL.Image.fromarray(im_offset.squeeze().mul(255).byte().numpy())

        return im_offset

    # align the images
    def get_images(self, image_set : ImageSetHdr):

        # get all the tensors and align them
        # note that we are not using the top1 channel right now
        # return a color image with different channels and a 
        # gray 3 channl image

        # create a dict to make life a little easier.
        im_dict = {}
        for im in image_set.images:
            im_dict[im.name] = im

        # convert to cv2
        images = [
             self.br.imgmsg_to_cv2(im_dict['bot1'].image),
             self.br.imgmsg_to_cv2(im_dict['bot2'].image),
             # self.br.imgmsg_to_cv2(detect_msg.image_top1),
             self.br.imgmsg_to_cv2(im_dict['top2'].image),
        ]

        # note the counts
        counts = [
            im_dict['bot1'].count,
            im_dict['bot2'].count,  
         #   detect_msg.count_top1,
            im_dict['top2'].count,
        ]

        # create the tensors from the images and counts
        im_color = torch.zeros(3, 500, 2464)
        im_color[0, :, :] = self.get_offset_image(images[0], 0)
        im_color[1, :, :] = self.get_offset_image(images[1], (counts[1] - counts[0]))
        im_color[2, :, :] = self.get_offset_image(images[2], (counts[2] - counts[0]))
        # im_tensors[3, :, :] = self.get_offset_image(images[3], (counts[3] - counts[0]))

        im_gray = self.get_red_image(im_color[2])
        
        return im_color, im_gray

                
    def get_sub_image(self, mask, box, im_tensor):
        #print(box)
        box_x1 = int(box[0])
        box_y1 = int(box[1])
        box_x2 = int(box[2])
        box_y2 = int(box[3])

        x1 = int((box_x1 + box_x2) / 2 - 64)
        x2 = x1 + 128
        y1 = int((box_y1 + box_y2) / 2 - 64)
        y2 = y1 + 128

        #print(mask.shape)
        #print(im_tensor.shape)
        masked_tensor = im_tensor * mask.squeeze()
        #print(masked_tensor.shape)
        #print(x1, x2, y1, y2)
        return masked_tensor[:, y1:y2, x1:x2]     
                

    def classify(self, prediction, img_tensor):

        #print(prediction)

        boxes = prediction['boxes']
        scores = prediction['scores']
        labels = prediction['labels']
        masks = prediction['masks']


        # ignore low scores

        mask_scores = scores.ge(0.4)
        #  was 0.5

        # width = boxes[:, 3] - boxes[:, 1]
        # height = boxes[:, 2] - boxes[:, 0]
        # mask_w = torch.logical_or(width.ge(150), width.le(10))
        # mask_h = torch.logical_or(height.ge(150), height.le(10))
        # mask_left = (boxes[:,0]).ge(125)
        # mask_right = (boxes[:,2]).le(2340)
        # mask_top = (boxes[:,1]).ge(65)
        # mask_bot = (boxes[:,3]).le(435)

        # mask_x = torch.logical_and(mask_left, mask_right)
        # mask_y = torch.logical_and(mask_top, mask_bot)

        # mask_xy = torch.logical_and(mask_x, mask_y)

        # mask = torch.logical_and(mask_scores, mask_xy)

        # boxes_masked = boxes[mask]
        # labels_masked = labels[mask]
        # scores_masked = scores[mask]

        pred_mask = torch.logical_and(boxes[:, 1].ge(64), boxes[:, 3].le(446))
        pred_mask = torch.logical_and(pred_mask, boxes[:, 0].ge(125))
        pred_mask = torch.logical_and(pred_mask, boxes[:, 2].le(2340))
        #print(scores.shape)
        #print(pred_mask.shape)
        pred_mask = torch.logical_and(pred_mask, scores.ge(0.75))

        boxes_masked = boxes[pred_mask]
        scores_masked = scores[pred_mask]
        masks_masked = masks[pred_mask]
        
        # mask_size = torch.logical_or(mask_w, mask_h)
        

        # mask = torch.logical_or(mask_size, mask)

        # ignore cherries outside region of interest
        broadcast_image_to_mask_start = time.time()
        
        # try:
        if torch.cuda.is_available():
            boxes_masked = boxes_masked.cuda()
            img_tensor = img_tensor.cuda()
            masks_masked = masks_masked.cuda()
        sub_tensors = torch.zeros([len(boxes_masked), 3, 128, 128])
        for argi in range (len(boxes_masked)):
            m = masks_masked[argi]
            b = boxes_masked[argi]
            # sub = (self.get_sub_image(m, b, img_tensor)).cuda()
            #print('shape sub', sub.shape)
            #print('shape sub aray', sub_tensors.shape)
            sub_tensors[argi] = (self.get_sub_image(m, b, img_tensor))
                
        # except Exception as e:
        #     print(e)
        #     sub_tensors = []


        if (len(sub_tensors) > 0):
            # sub_tensors = torch.stack(sub_tensors)

            broadcast_image_to_mask_end = time.time()
            print(f'broadcast mask; {broadcast_image_to_mask_end - broadcast_image_to_mask_start}')

            classify_start = time.time()

            #print(image_tensors.size())
            with torch.no_grad():
                if torch.cuda.is_available():
                    sub_tensors = sub_tensors.cuda()
                classifications = self.classifier(sub_tensors)
                #print(classifications)


            #print('classified')
            classify_end = time.time()
            print(f'do the classification; {classify_end - classify_start}')
            #print(classifications)

            classify_sort_start = time.time()

            probs = torch.nn.functional.softmax(classifications, dim=1)
            #print(probs)
            conf, classes = torch.max(probs, 1)

            #print('*************conf classes**************')
            #print(conf, classes)
            #print('*************conf classes**************')

            # [:, 0] -> get first collumn
            # [:, 0] -> get second collumn

            # get the pit, maybe, and clean categories
            # pit_mask = probs[:, 1].ge(self.pick_threshold)
            # maybe_mask = probs[:, 1].ge(self.maybe_threshold)
            # clean_mask = probs[:, 0].ge(1 - self.maybe_threshold)
            pit_mask = classes.eq (2)
            maybe_mask = classes.eq(1)
            clean_mask = classes.eq(0)

            # set the labels apropriately
            #print(classes)
            labels_masked = classes
            labels_masked = torch.where(maybe_mask, 5, labels_masked)
            labels_masked = torch.where(pit_mask, 2, labels_masked)
            labels_masked = torch.where(clean_mask, 1, labels_masked)

            #print('pit mask', pit_mask)
            # torch.where(pit_mask, labels, labels + 1  )
            # torch.where(maybe_mask, labels, labels + 4 )
            #torch.where(clean_mask, labels, labels  1  )

            # add the classification results to the prediction
            #prediction['labels'] = classes
            prediction['confidence'] = conf
            prediction['images'] = sub_tensors
            prediction['confidence_scores'] = classifications
            prediction['confidence_probs'] = probs

            prediction['boxes'] = boxes_masked
            prediction['scores'] = scores_masked
            prediction['labels'] = labels_masked

            #print('classified')
            classify_sort_end = time.time()
            print(f'sort for classify; {classify_sort_end - classify_sort_start}')
        else:
            prediction['confidence'] = []
            prediction['images'] = []
            prediction['confidence_scores'] = []
            prediction['confidence_probs'] = []

            prediction['boxes'] = []
            prediction['scores'] = []
            prediction['labels'] = []

        #print(classifications)

        return prediction
                

    def detect_stems(self, img_color):
    
        with torch.no_grad():
            stem_prediction = self.stem_model(img_color.unsqueeze(0).to(self.device))

        stem_prediction = stem_prediction[0]

        boxes = stem_prediction['boxes']
        scores = stem_prediction['scores']
        labels = stem_prediction ['labels']


        pred_mask = torch.logical_and(boxes[:, 1].ge(64), boxes[:, 3].le(446))
        pred_mask = torch.logical_and(pred_mask, boxes[:, 0].ge(125))
        pred_mask = torch.logical_and(pred_mask, boxes[:, 2].le(2340))
        #print(scores.shape)
        #print(pred_mask.shape)
        pred_mask = torch.logical_and(pred_mask, scores.ge(0.75))

        stem_prediction['boxes'] = boxes[pred_mask]
        stem_prediction['scores'] = scores[pred_mask]
        stem_prediction['labels'] = labels[pred_mask]

        return stem_prediction


    # get an detectionhdr messge
    def detect(self, detection_message):

        topil = T.ToPILImage()
        prep_for_seg_start = time.time()
        # make a copy of the image for labeling

        img_color, img_gray = self.get_images(detection_message)
        #print()

        #print(img_color.shape)
        img_labeled = img_color.clone().detach()

        # print('****************************  shape **********', img_labeled)
        ##  print('**************** img labled', img_labeled)
        #print('img_color shape: ', img_color.shape)
        #print('img_labeled shape: ', img_labeled.shape)

        predictions = None
        prep_for_seg_end = time.time()

        print(f'prep for start; {prep_for_seg_end - prep_for_seg_start}')

        prediction_start = time.time()

        # img_pil = Image.fromarray(np.uint8(img_gray*255))

        # img_pil.save('test_gray.png')
        # print(img_gray.shape)

        # img_gray = torch.tensor(img_gray)
        if torch.cuda.is_available():
            img_gray = img_gray.permute(2,0,1).cuda()
        else:
            img_gray = img_gray.permute(2,0,1)

        print('gray shape : {}'.format(img_gray.shape))
        print('color shape : {}'.format(img_color.shape))

        #print(img_gray.shape)

        # do the instance segmentation
        with torch.no_grad():
            predictions = self.model(img_gray.unsqueeze(0))

        prediction_end = time.time()
        print(f'predict; {prediction_end - prediction_start}')

        # this only processes 1 image, so get index 0
        prediction = predictions[0]

        stem_start = time.time()
        stem_prediction = self.detect_stems(img_color)
        stem_end = time.time()
        print(f'predict stems; {stem_end - stem_start}')
        

        #print(prediction['labels'])       
        filtered = {
            'boxes':[[]],
            'confidences': [[]],
            'labels': []
        }

        # only classify if the is 1 or more images to process 
        if len(prediction) > 0:
        
            # img_tensor = img_color.permute(2,0,1)
            # print(img_tensor.shape)

            # do the classification
            self.classify(prediction, img_color)

            # this filters things more

            #print(prediction)


            if (len(prediction['boxes']) > 0 ):
        #print('classified')

                # set cherries on side as '3 : side' label
                prediction['labels'] = torch.where(prediction['boxes'][:, 0] < 125, 3, prediction['labels'] )
                prediction['labels'] = torch.where(prediction['boxes'][:, 2] > 2340, 3, prediction['labels'] )

                sort_start = time.time()

                # transoform to go from tensor to PIL image
                

                # format the tensor iamge so that the draw_bounding_boxes function is happy
                img_for_labeling = (img_labeled * 255 )
                img_for_labeling = img_for_labeling.type(torch.uint8)
                

                clean_mask = prediction['labels'].eq(1)
                pit_mask = prediction['labels'].eq(2)
                side_mask = prediction['labels'].eq(3)
                maybe_mask = prediction['labels'].eq(5)



                cherry_found = len(prediction['boxes'])
                pit_found = len(prediction['boxes'][pit_mask])
                maybe_found = len(prediction['boxes'][maybe_mask])
                clean_found = len(prediction['boxes'][clean_mask])
                
                #print('image for albeling shape', img_for_labeling.shape)

                img_labeled_tensor = torchvision.utils.draw_bounding_boxes(img_for_labeling, prediction['boxes'][clean_mask], colors='limegreen', width=2)
                img_labeled_tensor = torchvision.utils.draw_bounding_boxes(img_labeled_tensor, prediction['boxes'][pit_mask], colors='red', width=2)
                img_labeled_tensor = torchvision.utils.draw_bounding_boxes(img_labeled_tensor, prediction['boxes'][side_mask], colors='cyan', width=2)
                img_labeled_tensor = torchvision.utils.draw_bounding_boxes(img_labeled_tensor, prediction['boxes'][maybe_mask], colors='yellow', width=2)
                img_labeled_tensor = torchvision.utils.draw_bounding_boxes(img_labeled_tensor, stem_prediction['boxes'], colors='black', width=2)
                img_labeled = topil(img_labeled_tensor)

                # convert tensor -> pil -> cv2
                img_labeled = np.array(img_labeled) 
                img_labeled = cv2.cvtColor(img_labeled, cv2.COLOR_RGB2BGR)


                # convert data to numpy arrays
                #masks = prediction['masks'].cpu().numpy()
                #scores = prediction['scores'].cpu().numpy()
                boxes = prediction['boxes'].cpu().numpy()
                labels = prediction['labels'].cpu().numpy()
                #confidences = prediction['confidence'].cpu().numpy()
                #images = prediction['images']
                #confidence_scores = prediction['confidence_scores'].cpu().numpy()
                confidence_probs = prediction['confidence_probs'].cpu().numpy()

                print(prediction['confidence_probs'])




                # create a dictionary with the results
                filtered = {
                    'boxes' : boxes,
                    'confidences' : confidence_probs,
                    'labels' : labels
                }



                sort_end = time.time()
                print(f'sort after processing; {sort_end - sort_start}')

            else:
                # img_labeled = (img_labeled * 255).type(torch.uint8)
                img_labeled = (img_labeled * 255).type(torch.uint8)
                
                img_labeled = img_labeled.permute(1,2,0).numpy()
                img_labeled = cv2.cvtColor(img_labeled, cv2.COLOR_RGB2BGR)



        stem_prediction['boxes'] = stem_prediction['boxes'].cpu().numpy()
        stem_prediction['labels'] = stem_prediction['labels'].cpu().numpy()
        stem_prediction['scores'] = stem_prediction['scores'].cpu().numpy()


                #print('cherries found {}; pit found  {}; maybe found {}; clean found {}'.format(cherry_found, pit_found, maybe_found, clean_found ))
        return filtered, img_labeled, stem_prediction



if __name__ == '__main__':

    package_share_directory = get_package_share_directory('cherry_detection')
    weight_path = os.path.join(package_share_directory,'seg_model_red_v1.pt')
    weights = torch.load(weight_path, map_location=torch.device('cuda'))
    weight_path2 = os.path.join(package_share_directory,'classification-2_26_2025-iter5.pt')
    weights2 = torch.load(weight_path2, map_location=torch.device('cuda'))
    weight_path_stems = os.path.join(package_share_directory,'stem_model_10_5_2024.pt')
    weights_stems = torch.load(weight_path_stems, map_location=torch.device('cuda'))


    my_detector =  ai_detector_class_3(weights, weights2, weights_stems)




    msg = get_test_msg('/home/user/threading_ws/src/cherry_detection/cherry_detection','10439811')


    filtered_results, img_labeled = my_detector.detect(msg)

    start_time = time.time()
    filtered_results, img_labeled = my_detector.detect(msg)
    filtered_results, img_labeled = my_detector.detect(msg)
    filtered_results, img_labeled = my_detector.detect(msg)
    filtered_results, img_labeled = my_detector.detect(msg)
    filtered_results, img_labeled = my_detector.detect(msg)
    filtered_results, img_labeled = my_detector.detect(msg)
    filtered_results, img_labeled = my_detector.detect(msg)
    filtered_results, img_labeled = my_detector.detect(msg)
    filtered_results, img_labeled = my_detector.detect(msg)
    filtered_results, img_labeled = my_detector.detect(msg)
    end_time = time.time()

    durr = (end_time - start_time)/10

    print('time per: ' ,durr)



    #cv2.imshow('detections', img_labeled)
   # img_pil = cv2.cvtColor(img_labeled, cv2.COLOR_RGB2BGR)
    # print(img_labeled.shape)
    to_pil = torchvision.transforms.ToPILImage()
    pil_image = to_pil ( img_labeled)

    pil_image.show()
    pil_image.save('test2.png')

    #cv2.waitKey()