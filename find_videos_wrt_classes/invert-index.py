import os,sys
import cPickle

CLASS_NAMES = ['background', 'airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car', 'cattle', 'dog', 'domestic_cat', 'elephant', 'fox', 'giant_panda', 'hamster', 'horse', 'lion', 'lizard', 'monkey', 'motorcycle', 'rabbit', 'red_panda', 'sheep', 'snake', 'squirrel', 'tiger', 'train', 'turtle', 'watercraft', 'whale', 'zebra']


if __name__ == '__main__':
	objs = cPickle.load(open(sys.argv[1], 'rb'))
	print (len(objs.keys()))

	classes = CLASS_NAMES[1:]
	
	class_video = {}
	for class_ in classes:
		class_video[class_] = {}
		class_video[class_]['videos'] = []
		class_video[class_]['num_of_bboxes'] = 0
	
	for key in objs.keys():
		video = objs[key]
		for obj in video.keys():
			class_video[obj]['videos'].append([key, video[obj]['num']])
			class_video[obj]['num_of_bboxes'] += video[obj]['num'] 

	# print (class_video)
	for key in class_video:
		print (key, len(class_video[key]['videos']), class_video[key]['num_of_bboxes'])
		class_video[key]['videos'].sort(key=lambda x: x[1])


	check_obj = sys.argv[2]
	# sort the videos:
	print (check_obj)
	print (len(class_video[check_obj]['videos']), class_video[check_obj]['num_of_bboxes'])
	for video in class_video[check_obj]['videos']:
		print video

	# sort the videos by number of boxes
	
	
	
	
	
		
	
	
