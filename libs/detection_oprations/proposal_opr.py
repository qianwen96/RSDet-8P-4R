# encoding: utf-8
from libs.configs import cfgs
from libs.box_utils import bbox_transform
from libs.box_utils import nms_rotate
import tensorflow as tf
import cv2
import numpy as np



def bbox_angle(bbox):
    x1=[]
    y1=[]
    w=[]
    h=[]
    theta=[]
    for i in range(len(bbox)):
        rect=[bbox[i][0],bbox[i][1],bbox[i][2],bbox[i][3],bbox[i][4],bbox[i][5],bbox[i][6],bbox[i][7]]
        box = np.int0(rect)
        box = box.reshape([4, 2])
        rect1 = cv2.minAreaRect(box)
        x1.append(rect1[0][0])
        y1.append(rect1[0][1])
        w.append(rect1[1][0])
        h.append(rect1[1][1])
        theta.append(rect1[2])

    
    return np.array(x1, np.float32),np.array(y1, np.float32),np.array(w, np.float32),np.array(h, np.float32),np.array(theta, np.float32)


def filter_detections(boxes, scores, is_training):
    """
    :param boxes: [-1, 4]
    :param scores: [-1, ]
    :param labels: [-1, ]
    :return:
    """
    if is_training:
        indices = tf.reshape(tf.where(tf.greater(scores, cfgs.VIS_SCORE)), [-1, ])
    else:
        indices = tf.reshape(tf.where(tf.greater(scores, cfgs.FILTERED_SCORE)), [-1, ])

    

    if cfgs.NMS:
        filtered_boxes = tf.gather(boxes, indices)
        filtered_scores = tf.gather(scores, indices)
        # print(filtered_boxes.shape)
        # exit()
        #这里需要从anchor调整成五个值输入

        # perform NMS
        # print(filtered_boxes.shape)
        pred_ctr_x,pred_ctr_y,pred_w,pred_h,pred_theta= tf.py_func(func=bbox_angle,
                                                                    inp=[filtered_boxes],
                                                                    Tout=[tf.float32, tf.float32, tf.float32,tf.float32,tf.float32]) 
       

        filtered_boxes=tf.transpose(tf.stack([pred_ctr_x,pred_ctr_y,pred_w,pred_h,pred_theta]))
        filtered_boxes=tf.reshape(filtered_boxes,[-1,5])
        # print("filter",filtered_boxes.shape)
        # exit()
        # if is_training else 1000,
        nms_indices = nms_rotate.nms_rotate(decode_boxes=filtered_boxes,
                                            scores=filtered_scores,
                                            iou_threshold=cfgs.NMS_IOU_THRESHOLD,
                                            max_output_size=100 if is_training else 1000,
                                            use_angle_condition=False,
                                            angle_threshold=15,
                                            use_gpu=False)

        # filter indices based on NMS
        indices = tf.gather(indices, nms_indices)

    # add indices to list of all indices
    # return indices
    return indices


def postprocess_detctions(rpn_bbox_pred, rpn_cls_prob, anchors, is_training):

    boxes_pred = bbox_transform.rbbox_transform_inv_2(boxes=anchors, deltas=rpn_bbox_pred)
    if cfgs.METHOD == 'H':
        x_c = (anchors[:, 2] + anchors[:, 0]) / 2
        y_c = (anchors[:, 3] + anchors[:, 1]) / 2
        w= anchors[:, 2] - anchors[:, 0] + 1
        h= anchors[:, 3] - anchors[:, 1] + 1
        anchors = tf.transpose(tf.stack([x_c, y_c, w, h]))

    return_boxes_pred = []
    return_scores = []
    return_labels = []
    for j in range(0, cfgs.CLASS_NUM):
        indices= filter_detections(boxes_pred, rpn_cls_prob[:, j], is_training)
        tmp_boxes_pred = tf.reshape(tf.gather(boxes_pred, indices), [-1, 8])#change from 5 to 8
        tmp_scores = tf.reshape(tf.gather(rpn_cls_prob[:, j], indices), [-1, ])

        return_boxes_pred.append(tmp_boxes_pred)
        # return_boxes_pred.append(indices)
        return_scores.append(tmp_scores)
        return_labels.append(tf.ones_like(tmp_scores)*(j+1))

    return_boxes_pred = tf.concat(return_boxes_pred, axis=0)
    return_scores = tf.concat(return_scores, axis=0)
    return_labels = tf.concat(return_labels, axis=0)

    return return_boxes_pred, return_scores, return_labels
