from lib.models import LaneATT
import torch
import cv2

lane_att = LaneATT()
anchors = lane_att.draw_anchors(640,360)
cv2.imshow("anchors", anchors)
cv2.waitKey(0)