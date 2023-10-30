from ultralytics import YOLO
import cv2
# import cvzone
import math
from sort import *
from utils import *
# import glob
# import pandas as pd
# import re
# import shutil


def main(image_path, model_weights):
    # Load the image
    raw_image = cv2.imread(image_path)
    small_images = segment_image(image_path, raw_image, small_image_width,
                                 small_image_height, overlap)
    # image_paths = glob.glob('./data/*.jpg')
    # Initialize YOLO model
    model = YOLO(model_weights)

    for small_image, x, y, filename in small_images:
        print('\nfile name:', filename)
        # all_detection_results = []
        all_detection_boxes = []
        all_detection_scores = []
        all_detection_clses = []
        right_detected_boxes = []
        right_confidence_scores = []
        right_cls_num = []
        below_detected_boxes = []
        below_confidence_scores = []
        below_cls_num = []
        lower_right_detected_boxes = []
        lower_right_confidence_scores = []
        lower_right_cls_num = []

        if x == 0 and y == 0:
            image, detection_results, detected_boxes, confidence_scores, cls_num = process_and_detect_image(
                filename, model, x, y)
            # all_detection_results.extend(detection_results)
            all_detection_boxes.extend(detected_boxes)
            all_detection_scores.extend(confidence_scores)
            all_detection_clses.extend(cls_num)

            if x + small_image_width < raw_image.shape[1]:
                x_right = x + small_image_width
                y_right = y
                right_image_path = generate_image_path(
                    filename, x_right, y_right)
                print('right name:', right_image_path)
                right_image, right_detection_results, right_detected_boxes, right_confidence_scores, right_cls_num = process_and_detect_image(
                    right_image_path, model, x_right, y_right)
                # all_detection_results.extend(right_detection_results)
                all_detection_boxes.extend(right_detected_boxes)
                all_detection_scores.extend(right_confidence_scores)
                all_detection_clses.extend(right_cls_num)

            if y + small_image_height < raw_image.shape[0]:
                x_below = x
                y_below = y + small_image_height
                below_image_path = generate_image_path(
                    filename, x_below, y_below)
                print('below name:', below_image_path)
                below_image, below_detection_results, below_detected_boxes, below_confidence_scores, below_cls_num = process_and_detect_image(
                    below_image_path, model, x_below, y_below)
                # all_detection_results.extend(below_detection_results)
                all_detection_boxes.extend(below_detected_boxes)
                all_detection_scores.extend(below_confidence_scores)
                all_detection_clses.extend(below_cls_num)

            if x + small_image_width < raw_image.shape[1] and y + small_image_height < raw_image.shape[0]:
                x_lower_right = x + small_image_width
                y_lower_right = y + small_image_height
                lower_right_image_path = generate_image_path(
                    filename, x_lower_right, y_lower_right)
                print('lower right name:', lower_right_image_path)
                lower_right_image, lower_right_detection_results, lower_right_detected_boxes, lower_right_confidence_scores, lower_right_cls_num = process_and_detect_image(
                    lower_right_image_path, model, x_lower_right, y_below)
                # all_detection_results.extend(lower_right_detection_results)
                all_detection_boxes.extend(lower_right_detected_boxes)
                all_detection_scores.extend(lower_right_confidence_scores)
                all_detection_clses.extend(lower_right_cls_num)

            # Apply NMS to remove duplicates across all photos
            # Use a global threshold for NMS
            boxes_to_keep = non_max_suppression(
                all_detection_boxes,  # Bounding boxes
                all_detection_scores,  # Confidence scores
                all_detection_clses,  # Class names
                global_nms_threshold
            )

            get_intersection(x, y, boxes_to_keep, detected_boxes, right_detected_boxes,
                             below_detected_boxes, lower_right_detected_boxes)

        if 0 < x < raw_image.shape[1] and y == 0:
            detected_boxes = [item[0]
                              for item in right_intersection[f'{x-small_image_width}_{y}']]
            confidence_scores = [item[1]
                                 for item in right_intersection[f'{x-small_image_width}_{y}']]
            cls_num = [item[2]
                       for item in right_intersection[f'{x-small_image_width}_{y}']]
            # all_detection_results.extend(detection_results)
            all_detection_boxes.extend(detected_boxes)
            all_detection_scores.extend(confidence_scores)
            all_detection_clses.extend(cls_num)

            if x + small_image_width < raw_image.shape[1]:
                x_right = x + small_image_width
                y_right = y
                right_image_path = generate_image_path(
                    filename, x_right, y_right)
                print('right name:', right_image_path)
                right_image, right_detection_results, right_detected_boxes, right_confidence_scores, right_cls_num = process_and_detect_image(
                    right_image_path, model, x_right, y_right)
                # all_detection_results.extend(right_detection_results)
                all_detection_boxes.extend(right_detected_boxes)
                all_detection_scores.extend(right_confidence_scores)
                all_detection_clses.extend(right_cls_num)

            if y + small_image_height < raw_image.shape[0]:
                below_detected_boxes = [item[0]
                                        for item in lower_right_intersection[f'{x-small_image_width}_{y}']]
                below_confidence_scores = [item[1]
                                           for item in lower_right_intersection[f'{x-small_image_width}_{y}']]
                below_cls_num = [item[2]
                                 for item in lower_right_intersection[f'{x-small_image_width}_{y}']]
                # all_detection_results.extend(below_detection_results)
                all_detection_boxes.extend(below_detected_boxes)
                all_detection_scores.extend(below_confidence_scores)
                all_detection_clses.extend(below_cls_num)

            if x + small_image_width < raw_image.shape[1] and y + small_image_height < raw_image.shape[0]:
                x_lower_right = x + small_image_width
                y_lower_right = y + small_image_height
                lower_right_image_path = generate_image_path(
                    filename, x_lower_right, y_lower_right)
                print('lower right name:', lower_right_image_path)
                lower_right_image, lower_right_detection_results, lower_right_detected_boxes, lower_right_confidence_scores, lower_right_cls_num = process_and_detect_image(
                    lower_right_image_path, model, x_lower_right, y_below)
                # all_detection_results.extend(lower_right_detection_results)
                all_detection_boxes.extend(lower_right_detected_boxes)
                all_detection_scores.extend(lower_right_confidence_scores)
                all_detection_clses.extend(lower_right_cls_num)

            # Apply NMS to remove duplicates across all photos
            # Use a global threshold for NMS
            boxes_to_keep = non_max_suppression(
                all_detection_boxes,  # Bounding boxes
                all_detection_scores,  # Confidence scores
                all_detection_clses,
                global_nms_threshold
            )
            # print(boxes_to_keep)

            get_intersection(x, y, boxes_to_keep, detected_boxes, right_detected_boxes,
                             below_detected_boxes, lower_right_detected_boxes)

        if x == 0 and 0 < y < raw_image.shape[0]:
            # print(below_intersection)
            detected_boxes = [item[0]
                              for item in below_intersection[f'{x}_{y-small_image_height}']]
            confidence_scores = [item[1]
                                 for item in below_intersection[f'{x}_{y-small_image_height}']]
            cls_num = [item[2]
                       for item in below_intersection[f'{x}_{y-small_image_height}']]
            # all_detection_results.extend(detection_results)
            all_detection_boxes.extend(detected_boxes)
            all_detection_scores.extend(confidence_scores)
            all_detection_clses.extend(cls_num)

            if x + small_image_width < raw_image.shape[1]:
                right_detected_boxes = [item[0]
                                        for item in lower_right_intersection[f'{x}_{y-small_image_height}']]
                right_confidence_scores = [item[1]
                                           for item in lower_right_intersection[f'{x}_{y-small_image_height}']]
                right_cls_num = [item[2]
                                 for item in lower_right_intersection[f'{x}_{y-small_image_height}']]
                # all_detection_results.extend(right_detection_results)
                all_detection_boxes.extend(right_detected_boxes)
                all_detection_scores.extend(right_confidence_scores)
                all_detection_clses.extend(right_cls_num)

            if y + small_image_height < raw_image.shape[0]:
                x_below = x
                y_below = y + small_image_height
                below_image_path = generate_image_path(
                    filename, x_below, y_below)
                print('below name:', below_image_path)
                below_image, below_detection_results, below_detected_boxes, below_confidence_scores, below_cls_num = process_and_detect_image(
                    below_image_path, model, x_below, y_below)
                # all_detection_results.extend(below_detection_results)
                all_detection_boxes.extend(below_detected_boxes)
                all_detection_scores.extend(below_confidence_scores)
                all_detection_clses.extend(below_cls_num)

            if x + small_image_width < raw_image.shape[1] and y + small_image_height < raw_image.shape[0]:
                x_lower_right = x + small_image_width
                y_lower_right = y + small_image_height
                lower_right_image_path = generate_image_path(
                    filename, x_lower_right, y_lower_right)
                print('lower right name:', lower_right_image_path)
                lower_right_image, lower_right_detection_results, lower_right_detected_boxes, lower_right_confidence_scores, lower_right_cls_num = process_and_detect_image(
                    lower_right_image_path, model, x_lower_right, y_below)
                # all_detection_results.extend(lower_right_detection_results)
                all_detection_boxes.extend(lower_right_detected_boxes)
                all_detection_scores.extend(lower_right_confidence_scores)
                all_detection_clses.extend(lower_right_cls_num)

            # Apply NMS to remove duplicates across all photos
            # Use a global threshold for NMS
            boxes_to_keep = non_max_suppression(
                all_detection_boxes,  # Bounding boxes
                all_detection_scores,  # Confidence scores
                all_detection_clses,
                global_nms_threshold
            )
            # print(boxes_to_keep)
            get_intersection(x, y, boxes_to_keep, detected_boxes, right_detected_boxes,
                             below_detected_boxes, lower_right_detected_boxes)

        if 0 < x < raw_image.shape[1] and 0 < y < raw_image.shape[0]:
            detected_boxes = [item[0]
                              for item in right_intersection[f'{x-small_image_width}_{y}']]
            confidence_scores = [item[1]
                                 for item in right_intersection[f'{x-small_image_width}_{y}']]
            cls_num = [item[2]
                       for item in right_intersection[f'{x-small_image_width}_{y}']]
            # all_detection_results.extend(detection_results)
            all_detection_boxes.extend(detected_boxes)
            all_detection_scores.extend(confidence_scores)
            all_detection_clses.extend(cls_num)

            if x + small_image_width < raw_image.shape[1]:
                x_right = x + small_image_width
                y_right = y
                right_image_path = generate_image_path(
                    filename, x_right, y_right)
                right_image, right_detection_results, right_detected_boxes, right_confidence_scores, right_cls_num = process_and_detect_image(
                    right_image_path, model, x_right, y_right)
                # all_detection_results.extend(right_detection_results)
                all_detection_boxes.extend(right_detected_boxes)
                all_detection_scores.extend(right_confidence_scores)
                all_detection_clses.extend(right_cls_num)

            if y + small_image_height < raw_image.shape[0]:
                below_detected_boxes = [item[0]
                                        for item in lower_right_intersection[f'{x-small_image_width}_{y}']]
                below_confidence_scores = [item[1]
                                           for item in lower_right_intersection[f'{x-small_image_width}_{y}']]
                below_cls_num = [item[2]
                                 for item in lower_right_intersection[f'{x-small_image_width}_{y}']]
                # all_detection_results.extend(below_detection_results)
                all_detection_boxes.extend(below_detected_boxes)
                all_detection_scores.extend(below_confidence_scores)
                all_detection_clses.extend(below_cls_num)

            if x + small_image_width < raw_image.shape[1] and y + small_image_height < raw_image.shape[0]:
                x_lower_right = x + small_image_width
                y_lower_right = y + small_image_height
                lower_right_image_path = generate_image_path(
                    filename, x_lower_right, y_lower_right)
                print('lower right name:', lower_right_image_path)
                lower_right_image, lower_right_detection_results, lower_right_detected_boxes, lower_right_confidence_scores, lower_right_cls_num = process_and_detect_image(
                    lower_right_image_path, model, x_lower_right, y_below)
                # all_detection_results.extend(lower_right_detection_results)
                all_detection_boxes.extend(lower_right_detected_boxes)
                all_detection_scores.extend(lower_right_confidence_scores)
                all_detection_clses.extend(lower_right_cls_num)

            # Apply NMS to remove duplicates across all photos
            # Use a global threshold for NMS
            boxes_to_keep = non_max_suppression(
                all_detection_boxes,  # Bounding cdboxes
                all_detection_scores,  # Confidence scores
                all_detection_clses,
                global_nms_threshold
            )
            # print(boxes_to_keep)
            get_intersection(x, y, boxes_to_keep, detected_boxes, right_detected_boxes,
                             below_detected_boxes, lower_right_detected_boxes)

    for class_name, count in object_counts.items():
        print(f"Total number of {class_name}: {count} ")


if __name__ == "__main__":
    input_image_path = 'data_raw/Screenshot 2023-10-30 104056.jpg'
    # input_image_path = 'Screenshot 2023-10-20 163357.jpg'

    model_weights = 'best-cls7-colab.pt'
    main(input_image_path, model_weights)
