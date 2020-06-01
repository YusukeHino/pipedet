from voc_eval import voc_eval, parse_rec, voc_eval_twoArrs
from mark_frame_with_bbox import annotate_image_with_bounding_boxes
import matplotlib.pyplot as plt
import numpy as np

file = "2to6_over20/My4K__S1000010_5fps_2to6"
gt = "_S1000010_5fps"
output_model_predictions_folder = "/media/ekmek/VitekDrive_I/___Results_for_internship/Accuracy/Custom_4k_videos/"+file+"/"
gt_path_folder = "/media/ekmek/VitekDrive_I/2017-18_CMU internship, part 1, Fall 2017/4K_DATASET_REC/annotations/samples/"+gt+"/"
globally_saved_annotations = "/media/ekmek/VitekDrive_I/2017-18_CMU internship, part 1, Fall 2017/4K_DATASET_REC/annotations/samples/"

image_redirection_path = "/media/ekmek/VitekDrive_I/2017-18_CMU internship, part 1, Fall 2017/4K_DATASET_REC/" \
                         "annotations/samples - SLOWLY blurring/S1000010_5fps/"

#image_redirection_path = gt_path_folder


show_figures = True
draw_text = False # text with confidences

imagesetfile = output_model_predictions_folder+"annotnames.txt"
predictions_file  = output_model_predictions_folder+"annotbboxes.txt"
rec, prec, ap = voc_eval(predictions_file,gt_path_folder,imagesetfile,'person')

print("ap", ap)

with open(predictions_file, 'r') as f:
    lines = f.readlines()
predictions = [x.strip().split(" ") for x in lines]

predictions_dict = {}

for pred in predictions:
    score = float(pred[1])
    # <image identifier> <confidence> <left> <top> <right> <bottom>
    left   = int(pred[2])
    top    = int(pred[3])
    right  = int(pred[4])
    bottom = int(pred[5])
    arr = [score, left, top, right, bottom]
    if not pred[0] in predictions_dict:
        predictions_dict[pred[0]] = []
    predictions_dict[pred[0]].append(arr)

print("predictions",len(predictions_dict), predictions_dict)

with open(imagesetfile, 'r') as f:
    lines = f.readlines()
imagenames = [x.strip() for x in lines]

aps = []

#imagenames = [imagenames[-1]]
print(imagenames)
imagenames = ['0077']
#imagenames = [imagenames[0]]

for imagename in imagenames:
    img = image_redirection_path + imagename + ".jpg"
    gt_file = gt_path_folder + imagename + ".xml"

    gt = parse_rec(gt_file)
    if imagename not in predictions_dict:
        continue
    predictions = predictions_dict[imagename ]

    print(imagename)
    print("ground truth:", len(gt), gt)
    print("predictions:", len(predictions), predictions)

    colors = [(0,128,0,125), (255, 165,0,125)] # green=GT orange=PRED
    colors = [(0,128,0,0), (255, 0,0,255)] # red=PRED
    colors = [(0,128,0,0), "lightcoral"] # red=PRED

    bboxes_gt = []
    bboxes_pred = []
    c_gt = 0
    for i in gt:
        bb = i["bbox"]
        print(bb)
        # left, top, right, bottom => top, left, bottom, right
        bb = [bb[1], bb[0], bb[3], bb[2]]
        bboxes_gt.append(['person_gt', bb, 1.0, c_gt])

    print("-")


    c_pred = 1
    for p in predictions:
        print(p)
        bb = p[1:]
        bb = [bb[1], bb[0], bb[3], bb[2]]
        bboxes_pred.append(['person', bb, p[0], c_pred])

    bboxes = bboxes_gt + bboxes_pred
    #bboxes = bboxes_gt
    bboxes = bboxes_pred

    rec, prec, ap = voc_eval_twoArrs(bboxes_gt,bboxes_pred,ovthresh=0.5)
    aps.append(ap)

    # show only not working ones:
    #show_figures = (ap < 1.0)

    if show_figures:
        img = annotate_image_with_bounding_boxes(img, "", bboxes, colors, ignore_crops_drawing=True, draw_text=draw_text,
                                       show=False, save=False, thickness=[0.0, 6.0], resize_output = 1.0)

    img.save("last_gt_pred.jpg", quality=90)

    if show_figures:
        #fig = plt.figure()
        plt.imshow(img)
        plt.title("Frame " + imagename + ", ap: " + str(ap) + " (green=GT orange=PRED)")
        plt.show()
        plt.clf()

    print("")

#print(aps)
#print("[AP] min, max, avg:",np.min(aps), np.max(aps), np.mean(aps))
#fig = plt.figure()
#plt.title("AP over frames, avg: "+str(np.mean(aps)))
#plt.plot(aps)
#plt.show()
