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



	vid_list = open(sys.argv[1])
	vid_objs = {}
	for video_name in vid_list:
		vid_name = video_name.strip()
		print video_name
	
		print (vid_name)
		## load vid tracks
		video_foler = VID_ROOT + vid_name
		vid = vid_proto_from_dir(video_foler, vid_name)
		print (len(vid['frames']))
	
		anno_folder = ANNO_ROOT + vid_name
		annot = get_anno(anno_folder)
		# print (annot.keys())
		# print (annot['video'])
		# print (len(annot['annotations']))

		# find fids that are needed
		fids = videos[vid_name]
		# print (fids)
	
		## match the gt with detection for each video
		objs = {}
		for frame_index, frame in enumerate(vid['frames']):
			# print (frame)
			# print (vid['root_path'])
			# imgpath = frame_path_at(vid, frame['frame'])
			# img = cv2.imread(imgpath)

			# load gt boundbox and annotation 
			# print (annot['annotations'])
		        boxes = [track_box_at_frame(tracklet, frame['frame']) for tracklet in [anno['track'] for anno in annot['annotations']]]
	        	classes = [track_class_at_frame(tracklet, frame['frame']) for tracklet in [anno['track'] for anno in annot['annotations']]]
			# print (boxes)
			# print (classes)

			for bbox_, class_ in zip(boxes, classes):
			 	if bbox_ != None and class_ != None:
					# print bbox_, class_	
					if class_ not in objs.keys():
						objs[class_] = {}
						objs[class_]['num'] = 0
					objs[class_]['num'] += 1
		# print objs
		vid_objs[vid_name] = objs
		# if len(vid_objs.keys()) == 10:
		# 	break
	print vid_objs
	import cPickle
	savefile = 'video_class_num.pkl'
	cPickle.dump(vid_objs, open(savefile, 'wb'))

	
	
	

	

	
