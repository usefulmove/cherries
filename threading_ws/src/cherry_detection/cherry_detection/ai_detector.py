
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
# from engine import train_one_epoch, evaluate
import torchvision.transforms as T
import cv2
from PIL import Image
from datetime import datetime

import random

import matplotlib.pyplot as plt
from ament_index_python.packages import get_package_share_directory
from torchvision.models import resnet50 #, ResNet50_Weights
import json

import time
#import functorch





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
    print(classification)
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

class ai_detector_class:
    def __init__(self, weights, weights2):

        # get the device type
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.pick_threshold = 0.5
        self.maybe_threshold = 0.25

        # get the weights from the package share directory
        #package_share_directory = get_package_share_directory('cherry_detection')
        #weight_path = os.path.join(package_share_directory, 'segmentation_20.pt')
        #weight_path = 'segmentation_20.pt'

        #weights = torch.load(weight_path)

        # get the model definition, load weights, and set to eval mode, move to gpu
        # we have two classes, 0: background, 1: cherry matter
        self.model = self.get_instance_segmentation_model(2)
        self.model.load_state_dict(weights)
        self.model.eval()
        self.model.to(self.device )

        # define some transforms
        self.tesnor_to_img = T.PILToTensor() 



        self.classifier = resnet50().to(self.device)
        num_ftrs = self.classifier.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        self.classifier.fc = torch.nn.Linear(num_ftrs, 2)

        self.classifier.load_state_dict(weights2)
        self.classifier.eval()
        self.classifier.cuda()


        self.count_im = 0
        self.count_cherry = 0


    # HxWxC 8bit [0 , 255] to CxHxW float [0., 1.]
    # nd_array or PIL image
    # to tensor converts HxWxC 8bit 0-225 to CxHxW float 0.-1.
    def get_instance_segmentation_model(self, num_classes):
    # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=False, 
            num_classes=2,

            # base class MaskR-CNN parameters
            # transform parameters
            min_size=800,
            max_size=2464,
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
            rpn_batch_size_per_image=256,
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



    def classify(self, prediction, img_tensor):

        # these are tensors
        masks = prediction['masks']
        #print(masks)
        #scores = prediction['scores']
        boxes = prediction['boxes']
        #labels = prediction['labels']
        #confidence = np.zeros(len(scores))

        broadcast_image_to_mask_start = time.time()

        # mask is [x, 1, 500, 2463]
        # image is [3, 500, 2463]
        # we cannot broadcast directly because the iamge dimensinos are not equal ( 3 vs 1)
        # we want to try and do as much math as possible on the gpu to speed things up

        size_masks = masks.size()

        pad_im_transform = T.transforms.CenterCrop(128)

        # kernel used for dialate - erode image processing
        kernel = np.array([ [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1] ], dtype=np.float32)


        kernel_tensor = torch.tensor(np.expand_dims(np.expand_dims(kernel, 0), 0),  device='cuda') # size: (1, 1, 5, 5)
        

        # create a destination array for the image to process
        cl_imgs = torch.zeros(size_masks[0],3,128,128, device='cuda')

        # dialate erode each mask
        # use mask to create 128x128 images
        for index, mask in enumerate(masks):
            cl_img = torch.zeros(3,128,128, device='cuda')
            bbox = boxes[index].type(torch.int) 
            #print(bbox)

            #cl_img =  (img_tensor)[0:3,bbox[1]:bbox[3],bbox[0]:bbox[2]]
            
            #print(f'cl_img_size = {cl_img.size()}')
            #print(cl_img)

            mask = (masks[index])[0:1,bbox[1]:bbox[3],bbox[0]:bbox[2]]
            mask = torch.clamp(torch.nn.functional.conv2d(mask, kernel_tensor, padding=(1, 1)), 0, 1)
            mask = torch.clamp(torch.nn.functional.conv2d(mask, kernel_tensor * -1, padding=(1, 1)), 0, 1)

            cl_img = ((masks[index])[0:1,bbox[1]:bbox[3],bbox[0]:bbox[2]]) * (img_tensor)[0:3,bbox[1]:bbox[3],bbox[0]:bbox[2]]

            cl_imgs[index] = pad_im_transform(cl_img)


        #    ****   test the dialte erode and image segmentation *********
        # topil = T.ToPILImage()
        # count = 0
        # for climg in cl_imgs:
        #     climg_pil = topil(climg)
        #     #print(climg_pil)
        #     # climg_pil_cpu = climg_pil.cpu()
        #     climg_pil_numpy = np.array(climg_pil) 
        #     climg_cv2 = cv2.cvtColor(climg_pil_numpy, cv2.COLOR_RGB2BGR)
        #     cv2.imwrite(f'cl_img_{count}.png', climg_cv2)
        #     count = count + 1
        # #print(cl_imgs)

        #    ****   test the dialte erode and image segmentation *********

        broadcast_image_to_mask_end = time.time()
        print(f'broadcast mask; {broadcast_image_to_mask_end - broadcast_image_to_mask_start}')

        classify_start = time.time()

        #print(image_tensors.size())
        with torch.no_grad():
            classifications = self.classifier(cl_imgs)
            #print(classifications)

        #print('classified')
        classify_end = time.time()
        print(f'do the classification; {classify_end - classify_start}')
        #print(classifications)

        classify_sort_start = time.time()

        probs = torch.nn.functional.softmax(classifications, dim=1)
        #print(probs)
        conf, classes = torch.max(probs, 1)

        #print(conf, classes)

        # [:, 0] -> get first collumn
        # [:, 0] -> get second collumn

        # get the pit, maybe, and clean categories
        pit_mask = probs[:, 1].ge(self.pick_threshold)
        maybe_mask = probs[:, 1].ge(self.maybe_threshold)
        clean_mask = probs[:, 0].ge(1 - self.maybe_threshold)

        # set the labels apropriately
        prediction['labels'] = torch.where(maybe_mask, 5, prediction['labels'] )
        prediction['labels'] = torch.where(pit_mask, 2, prediction['labels'] )
        prediction['labels'] = torch.where(clean_mask, 1, prediction['labels'] )

        #print('pit mask', pit_mask)
        # torch.where(pit_mask, labels, labels + 1  )
        # torch.where(maybe_mask, labels, labels + 4 )
        #torch.where(clean_mask, labels, labels  1  )

        # add the classification results to the prediction
        #prediction['labels'] = classes
        prediction['confidence'] = conf
        prediction['images'] = cl_imgs
        prediction['confidence_scores'] = classifications
        prediction['confidence_probs'] = probs

        #print('classified')
        classify_sort_end = time.time()
        print(f'sort for classify; {classify_sort_end - classify_sort_start}')
        #print(classifications)

        return prediction
                





    def detect(self, img_cv2):

        # crop the image to the reelevant section
        img_cv2 = img_cv2[200:700, 0:2463]

        prep_for_seg_start = time.time()
        # make a copy of the image for labeling
        img_labeled = img_cv2.copy()

        # move the image to a tensor on the GPU
        img_pil = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
        img_tensor = pil_to_tensor_gpu(img_pil, self.device)

        predictions = None
        prep_for_seg_end = time.time()

        print(f'prep for start; {prep_for_seg_end - prep_for_seg_start}')

        prediction_start = time.time()

        # do the instance segmentation
        with torch.no_grad():
            predictions = self.model(img_tensor.unsqueeze(0))

        prediction_end = time.time()
        print(f'predict; {prediction_end - prediction_start}')

        # this only processes 1 image, so get index 0
        prediction = predictions[0]



        #mask_location_left = prediction['boxes'][:, 0] < 174 
        #mask_location_right = prediction['boxes'][:, 3] > 2074


        #print(mask_location_left)
        #print(mask_location_right)
        #print(labels)



        #print(prediction['labels'])       
        filtered = []  
        # only classify if the is 1 or more images to process 
        if len(prediction) > 0:
        
            # do the classification
            self.classify(prediction, img_tensor.squeeze())
        #print('classified')

            # set cherries on side as '3 : side' label
            prediction['labels'] = torch.where(prediction['boxes'][:, 0] < 170, 3, prediction['labels'] )
            prediction['labels'] = torch.where(prediction['boxes'][:, 2] > 2244, 3, prediction['labels'] )

            sort_start = time.time()

            # transoform to go from tensor to PIL image
            topil = T.ToPILImage()

            # format the tensor iamge so that the draw_bounding_boxes function is happy
            img_for_labeling = (img_tensor * 255 )
            img_for_labeling = img_for_labeling.type(torch.uint8)


            clean_mask = prediction['labels'].eq(1)
            pit_mask = prediction['labels'].eq(2)
            side_mask = prediction['labels'].eq(3)
            maybe_mask = prediction['labels'].eq(5)

            cherry_found = len(prediction['boxes'])
            pit_found = len(prediction['boxes'][pit_mask])
            maybe_found = len(prediction['boxes'][maybe_mask])
            clean_found = len(prediction['boxes'][clean_mask])


            img_labeled_tensor = torchvision.utils.draw_bounding_boxes(img_for_labeling, prediction['boxes'][clean_mask], colors='limegreen', width=2)
            img_labeled_tensor = torchvision.utils.draw_bounding_boxes(img_labeled_tensor, prediction['boxes'][pit_mask], colors='red', width=2)
            img_labeled_tensor = torchvision.utils.draw_bounding_boxes(img_labeled_tensor, prediction['boxes'][side_mask], colors='cyan', width=2)
            img_labeled_tensor = torchvision.utils.draw_bounding_boxes(img_labeled_tensor, prediction['boxes'][maybe_mask], colors='yellow', width=2)
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


            # create a dictionary with the results
            filtered = {
                'boxes' : boxes,
                'confidences' : confidence_probs,
                'labels' : labels
            }



            sort_end = time.time()
            print(f'sort after processing; {sort_end - sort_start}')

            print('cherries found {}; pit found  {}; maybe found {}; clean found {}'.format(cherry_found, pit_found, maybe_found, clean_found ))
            # print(f'cherries found {len(filtered)}')

        return filtered, img_labeled


if __name__ == '__main__':

    package_share_directory = get_package_share_directory('cherry_detection')
    weight_path = os.path.join(package_share_directory,'cherry_segmentation.pt')
    weights = torch.load(weight_path)
    weight_path2 = os.path.join(package_share_directory,'cherry_classification.pt')
    weights2 = torch.load(weight_path2)

    my_detector =  ai_detector_class(weights, weights2)

    # img = cv2.imread('/home/user/Pictures/lots/image_20221102T155902.png')
    img = cv2.imread('/home/user/Pictures/12_6/bag_4/image_20221206T114617.png')

    #img = cv2.imread('/home/user/Pictures/cleans/clean_4.png')
    #img = cv2.imread('/home/user/Pictures/cherries/image_20221027T125232.png')
    #img = cv2.imread('/home/user/Pictures/cherries/small.png')

    #img = cv2.imread('/media/user/FADC0612DC05CA39/traina/instance_segmentation/task_segment-2022_10_19_18_25_33-coco 1.0/images/20220919T093001_color.jpg')
    #img = cv2.imread('/media/user/FADC0612DC05CA39/traina/instance_segmentation/task_segment-2022_10_19_18_25_33-coco 1.0/images/20220919T100642_color.jpg')
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    detections, img_labeled = my_detector.detect(img)

    print(detections)



    #cv2.imshow('detections', img_labeled)
    img_pil = cv2.cvtColor(img_labeled, cv2.COLOR_RGB2BGR)
    pil_image = Image.fromarray(img_pil)

    pil_image.show()

    pil_image.save('test.png')


    #cv2.waitKey()