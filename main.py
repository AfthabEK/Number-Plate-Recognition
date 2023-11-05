import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr


import util

# define constants
# define constants
model_cfg_path = os.path.join('.', 'model', 'cfg', 'darknet-yolov3.cfg')
model_weights_path = os.path.join('.', 'model', 'weights', 'model.weights')
class_names_path = os.path.join('.', 'model', 'class.names')

#input_dir = '/home/afthab/development/automatic-number-plate-recognition-python/data'
#input dir = local path to image
input_dir = './data'

# load class names
with open(class_names_path, 'r') as f:
    class_names = [j.strip() for j in f.readlines() if len(j) > 2]

# load model
net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)

# create EasyOCR reader
reader = easyocr.Reader(['en'])

for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)

    if not os.path.isfile(img_path):
        continue

    img = cv2.imread(img_path)

    H, W, _ = img.shape


    # convert image
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True)

    # get detections
    net.setInput(blob)

    detections = util.get_outputs(net)

    # bboxes, class_ids, confidences
    bboxes = []
    class_ids = []
    scores = []

    for detection in detections:
        # [x1, x2, x3, x4, x5, x6, ..., x85]
        bbox = detection[:4]

        xc, yc, w, h = bbox
        bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

        bbox_confidence = detection[4]

        class_id = np.argmax(detection[5:])
        score = np.amax(detection[5:])

        bboxes.append(bbox)
        class_ids.append(class_id)
        scores.append(score)

    # apply nms
    bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)
    text2 = ''
    # plot
    reader = easyocr.Reader(['en'])
    for bbox_, bbox in enumerate(bboxes):
        xc, yc, w, h = bbox



        license_plate = img[int(yc - (h / 2)):int(yc + (h / 2)), int(xc - (w / 2)):int(xc + (w / 2)), :].copy()

        img = cv2.rectangle(img,
                            (int(xc - (w / 2)), int(yc - (h / 2))),
                            (int(xc + (w / 2)), int(yc + (h / 2))),
                            (0, 255, 0),
                            15)

        license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)

        _, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

        output = reader.readtext(license_plate_thresh)
        
        text3=''
        conf=0.0
        for out in output:
            text_bbox, text, text_score = out
            if text_score > 0:
                print(text, text_score)
                #write to result.txt
                text2 = text2 + text

                conf=text_score
        #write to result.txt
        with open('result.txt', 'a') as f:
            f.write(text2 + ' ' + str(conf) + '\n')

    cv2.rectangle(img,
                  (int(xc - (w / 2)), int(yc - (h / 2))),
                  (int(xc + (w / 2)), int(yc + (h / 2))),
                  (0, 255, 0),
                  15)

    text = f"{class_names[class_ids[bbox_]]}: {text2} {conf}"
    font_scale = 0.5 * (w / 100)  # Scale the font with the width of the box
    font_thickness = int(2 * (w / 100))  # Adjust the font thickness as needed
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
    text_x = int(xc - (w / 2))
    text_y = int(yc - (h / 2)) - 40  # Adjust the vertical position as needed

    cv2.putText(img,
                text2 ,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 255, 0),
                font_thickness)
                
    #cv2.imshow('img', img)



    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # plt.figure()
    # plt.imshow(cv2.cvtColor(license_plate, cv2.COLOR_BGR2RGB))

    # plt.figure()
    # plt.imshow(cv2.cvtColor(license_plate_gray, cv2.COLOR_BGR2RGB))

    # plt.figure()
    # plt.imshow(cv2.cvtColor(license_plate_thresh, cv2.COLOR_BGR2RGB))

    plt.show()

    #SHOW IMAGE 

