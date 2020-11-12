

video 2 frames
```
ffmpeg -i 20200918_002.mp4 ./frames/%04d.jpg
```


frames 2 video
```
ffmpeg -r:v 60 -i "%04d.jpg" -codec:v libx264 -an "5_5_iou.mp4"
```
```
ffmpeg -r:v 60 -i "%04d.png" -codec:v libx264 -an "20200918_038_depth.mp4"
```

```
ffmpeg -r:v 60 -pattern_type glob -i "*.jpg" -codec:v libx264 -an "depth_demo.mp4"
```

```
ffmpeg -r:v 30 -pattern_type glob -i "*.png" -codec:v libx264 -r 30 "20200918_038_depth.mp4"
```


60 fps video 2 30 fps video
```
ffmpeg -i 5_5_iou.mp4 -vf fps=30 -codec:v libx264 "5_5_iou_30fps.mp4"
```


```
ffmpeg -r:v 60 -pattern_type glob -i "*.jpg" -codec:v libx264 -an "depth_mirror_114_335.mp4"
```

