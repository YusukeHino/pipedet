


frames 2 video
```
ffmpeg -r:v 60 -i "%04d.jpg" -codec:v libx264 -an "video.mp4"
```


60 fps video 2 30 fps video
```
ffmpeg -i video.mp4 -vf fps=30 -codec:v libx264 "video_30fps.mp4"
```