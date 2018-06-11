# VID_visualization


Visualization of IMAGENET VID dataset

# Usage

1. Find vidoes of one specific class, e.g. monkey.
```bash
cd find_videos_wrt_classes/
python invert-index.py video_class_num.pkl monkey
```

Results will be like

```bash
...
monkey
(19, 10654)
['ILSVRC2015_val_00170003', 106]
['ILSVRC2015_val_00170000', 110]
['ILSVRC2015_val_00135000', 112]
['ILSVRC2015_val_00127000', 120]
['ILSVRC2015_val_00078000', 149]
...
```


2. Get detection results of one video, e.g. ILSVRC2015_val_00170000, from one result,e.g. det_remove_lowschores-0.1.


```bash
cd ..
cd draw_gt_det
python save_gt_det.py det_remove_lowschores-0.1 ILSVRC2015_val_00170000
```



