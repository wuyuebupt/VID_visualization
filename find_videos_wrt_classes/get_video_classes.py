import os,sys
import cv2
from collections import defaultdict
from easydict import EasyDict

from vdetlib.utils.protocol import vid_proto_from_dir, frame_path_at
from vdetlib.tools.imagenet_annotation_processor import get_anno

VID_ROOT = '/home/yue/project/vid/code/videoVisualization/myVisual/ILSVRC2015/Data/VID/val/'
ANNO_ROOT = '/home/yue/project/vid/code/videoVisualization/myVisual/ILSVRC2015/Annotations/VID/val/'
IMAGESET_ROOT = '/home/yue/project/vid/code/videoVisualization/ILSVRC/ImageSets/VID/'

# RGB -> BGR

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



if __name__ == '__main__':

	videos = defaultdict(list)
	with open(os.path.join(IMAGESET_ROOT, 'val.txt')) as f:
		lines = f.readlines()
		for line in lines:
			line = line.strip().split()
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

	
	
	

	

	
