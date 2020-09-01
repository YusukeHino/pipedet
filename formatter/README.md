# for AutoML

```
cd ~/data/for_rsm_detection/cropped_1024_1024

gsutil cp {train,validate,test}/*.{png,JPG} gs://mirror-images/img/20200614_1024_1024
```

```
cd ~/data/for_rsm_detection/integrated_cropped_1024_1024

gsutil cp rsm_labels_1024_1024_for_automl.csv gs://mirror-csv/20200614_1024_1024_2/rsm_labels_1024_1024_for_automl.csv
```

```
gsutil rm gs://mirror-images/img/*_{0,1,2,3,4}.{png,JPG}
```