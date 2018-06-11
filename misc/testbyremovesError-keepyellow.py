import os,sys
import cv2
from collections import defaultdict
from easydict import EasyDict

from vdetlib.utils.protocol import vid_proto_from_dir, frame_path_at
from vdetlib.tools.imagenet_annotation_processor import get_anno

VID_ROOT = '/home/yue/project/vid/code/videoVisualization/myVisual/ILSVRC2015/Data/VID/val/'
ANNO_ROOT = '/home/yue/project/vid/code/videoVisualization/myVisual/ILSVRC2015/Annotations/VID/val/'
IMAGESET_ROOT = '/home/yue/project/vid/code/videoVisualization/ILSVRC/ImageSets/VID/'

CLASS_NAMES = ['background', 'airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car', 'cattle', 'dog', 'domestic_cat', 'elephant', 'fox', 'giant_panda', 'hamster', 'horse', 'lion', 'lizard', 'monkey', 'motorcycle', 'rabbit', 'red_panda', 'sheep', 'snake', 'squirrel', 'tiger', 'train', 'turtle', 'watercraft', 'whale', 'zebra']

PRESET_COLORS = [
  (255, 65, 54),
  (61, 153, 112),
  (0, 116, 217),
  (133, 20, 75),
  (0, 31, 63),
  (240, 18, 190),
  (1, 255, 112),
  (127, 219, 255),
  (255, 133, 27),
  (176, 176, 176)
]

## some colors
# RGB -> BGR
COLORS = {
	'green' : (0,255,127),
	'yellow': (0,215,255),
	'red':(0,0,255),
	'blue': (225,0,0),
	'gray': (85,85,85)
}
COLORS = EasyDict(COLORS)
FontColor = (0,0,0)


def track_class_at_frame(tracklet, frame_id):
    for box in tracklet:
        if box['frame'] == frame_id:
            return box['class']
    return None


def track_box_at_frame(tracklet, frame_id):
    for box in tracklet:
        if box['frame'] == frame_id:
            return box['bbox']
    return None


import numpy as np

def get_gt_thres(gt_bbox):
	# get gt_threshold -> for small objects
	gt_w = gt_bbox[2] - gt_bbox[0] + 1
	gt_h = gt_bbox[3] - gt_bbox[1] + 1
	thres = (gt_w * gt_h)/((gt_w + 10.) * (gt_h + 10.))
	gt_thr = np.min((0.5, thres))
	return gt_thr

def cal_IoU(bbox1, bbox2):
	# calculate real IoU
	bi = [np.max((bbox1[0], bbox2[0])), np.max((bbox1[1], bbox2[1])), np.min((bbox1[2], bbox2[2])),  np.min((bbox1[3], bbox2[3]))]
	iw = bi[2] - bi[0] + 1
	ih = bi[3] - bi[1] + 1
	if iw > 0 and ih > 0:
		ua = (bbox2[2] - bbox2[0] + 1.) * (bbox2[3] - bbox2[1] + 1.) + (bbox1[2] - bbox1[0] +1.)* (bbox1[3] - bbox1[1] + 1.) - iw * ih
		ov = iw * ih / ua
	else:
		ov = 0.
	return ov 
	



if __name__ == '__main__':

	## load all detection results
	
	det_file = sys.argv[1]
	videos = defaultdict(list)
	fid_to_path = {}
	with open(os.path.join(IMAGESET_ROOT, 'val.txt')) as f:
		lines = f.readlines()
		for line in lines:
			line = line.strip().split()
			fid_to_path[int(line[1])] = os.path.join(VID_ROOT, 'val', line[0] + '.JPEG')
			videos[os.path.dirname(line[0])].append(int(line[1]))

	# sort frames inside each video
	for k in videos:
		videos[k].sort()

	fid_to_video = {}
	with open(os.path.join(IMAGESET_ROOT, 'val.txt')) as f:
		lines = f.readlines()
		for line in lines:
			line = line.strip().split()
			fid_to_video[int(line[1])] = os.path.dirname(line[0])
			# videos[os.path.dirname(line[0])].append(int(line[1]))
			# videos[os.path.dirname(line[0])].append(int(line[1]))
	print (len(fid_to_video))
	
	output = open(sys.argv[2], 'w')
	
	video_name = -1
	with open(det_file, 'r') as f:
		lines = f.readlines()
		for line0 in lines:
			line = line0.strip().split()
			# v2: use order infor
			# print (fids)
			item = {
				'fid': int(line[0]),
				'class_index': int(line[1]),
				'score': float(line[2]),
				'bbox': map(float, line[3:])
			}
			item = EasyDict(item)
			pred = item
			# print (item)
			current_video_name = fid_to_video[int(line[0])]
			# get all labels when a new video comes
			if video_name != current_video_name:
				# load new video 
				video_name = current_video_name
				print ("load video: ", video_name)
				anno_folder = ANNO_ROOT + video_name
				annot = get_anno(anno_folder)
			        # video_foler = VID_ROOT + video_name
			        # vid = vid_proto_from_dir(video_foler, video_name)
				fids = videos[video_name]
			# print (int(line[0]), vid['frames'][0])
			frame = int(line[0]) - fids[0] + 1
			# print (frame)
	        	bboxes = [track_box_at_frame(tracklet, frame) for tracklet in [anno['track'] for anno in annot['annotations']]]
		        classes = [track_class_at_frame(tracklet, frame) for tracklet in [anno['track'] for anno in annot['annotations']]]

			ovmax = -1
			kmax = -1
			count = 0
			keytype = -1
			for bbox_, class_ in zip(bboxes, classes):
				if bbox_ != None and class_ != None:
					# draw the GT with yellow
					gt_thr = get_gt_thres(bbox_)
	
					# print pred, bbox_
					# calculate the IoU of two box
					IoU = cal_IoU(bbox_, pred.bbox)
					# print IoU, gt_thr
					# print pred.class_index, CLASS_NAMES[pred.class_index], class_
					
					# case 1 : correct detection, 0.5 <= IoU       + correct class label
					# case 2 : Error 1,             0 <  IoU < 0.5 + correct class label
					# case 3 : Error 2,           0.5 <= IoU       + wrong class label
					# case 4 : Error 3,             0 <  IoU < 0.5 + wrong class label
					pred_class_ = CLASS_NAMES[pred.class_index]
					if IoU >= gt_thr and class_ == pred_class_:
						# case 1
						if IoU >= ovmax:
							ovmax = IoU
							kmax = count
							count += 1
							keytype = 1
					elif IoU > 0.0 and IoU < gt_thr and class_ == pred_class_ and keytype != 1:
						# case 2
						
						keytype = 2
					elif IoU >= gt_thr and class_ != pred_class_ and keytype != 1:
						# case 3

						keytype = 3
					elif IoU > 0.0 and IoU < gt_thr and class_ != pred_class_ and keytype != 1:
						# case 3

						keytype = 4
					elif keytype != 1:
						keytype = 5
					else:
						keytype = 1 

			if keytype == 1:
				output.write(line0)
			elif keytype == 2:
				continue
				output.write(line0)
			elif keytype == 3:
				continue
				output.write(line0)
			elif keytype == 4:
				continue
				output.write(line0)
			elif keytype == 5:
				continue
				output.write(line0)

		output.close()	
		exit()

