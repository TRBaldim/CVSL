import cv2
import skvideo.io
import numpy as np


def diff_up_down(img):
    height, width, depth = img.shape
    half = height / 2
    top = img[0:half, 0:width]
    bottom = img[half:half+half, 0:width]
    top = cv2.flip(top, 1)
    bottom = cv2.resize(bottom, (32, 64))
    top = cv2.resize(top, (32, 64))
    return mse(top, bottom)


def diff_left_right(img):
    height, width, depth = img.shape
    half = width / 2
    left = img[0:height, 0:half]
    right = img[0:height, half:half+half-1]
    right = cv2.flip(right, 1)
    left = cv2.resize(left, (32, 64))
    right = cv2.resize(right, (32, 64))
    return mse(left, right)


def mse(image_a, image_b):
    '''
    Mean square error function
    '''
    err = np.sum((image_a.astype("float") - image_b.astype("float")) ** 2)
    err /= float(image_a.shape[0] * image_a.shape[1])
    return err


def is_new_roi(rx, ry, rw, rh, rectangles):
    for r in rectangles:
        if abs(r[0] - rx) < 40 and abs(r[1] - ry) < 40:
            return False
    return True


def detection_regions_of_interest(frame, cascade):

    frame_h, frame_w, f_depth = frame.shape
    print frame_h, frame_w
    scale_down = int(frame_w / 320)

    frame = cv2.resize(frame, (frame_w / scale_down, frame_h / scale_down))
    frame_h, frame_w, f_depth = frame.shape
    print frame_h, frame_w

    cars = cascade.detectMultiScale(frame, 1.2, 1)

    new_regions = []

    min_y = int(frame_h * 0.3)

    for (x, y, w, h) in cars:
        roi = [x, y, w, h]
        roi_image = frame[y:y+h, x:x+w]

        car_w = roi_image.shape[0]

        if y > min_y:
            diff_x = diff_left_right(roi_image)
            diff_y = round(diff_up_down(roi_image))

            if diff_x > 1600 and diff_x < 3000 and diff_y > 12000:
                rx, ry, rw, rh = roi
                new_regions.append([rx*scale_down, ry*scale_down, rw*scale_down, rh*scale_down])
    return new_regions


def detect_cars(file_path, classifier_path):
    rectangles = []
    cascade = cv2.CascadeClassifier(classifier_path)

    vc = skvideo.io.VideoCapture(file_path)

    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    roi = [0, 0, 0, 0]
    frame_count = 0

    while rval:
        rval, frame = vc.read()
        try:
            frame_h, frame_w, f_depth = frame.shape

            new_regions = detection_regions_of_interest(frame, cascade)

            for region in new_regions:
                if is_new_roi(region[0], region[1], region[2], region[3], rectangles):
                    rectangles.append(region)

            for r in rectangles:
                cv2.rectangle(frame, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (0, 0, 255), 3)

            frame_count += 1

            if frame_count > 30:
                frame_count = 0
                rectangles = []

            cv2.imshow("Result", frame)
            cv2.waitKey(1)
        except:
            break
    vc.release()