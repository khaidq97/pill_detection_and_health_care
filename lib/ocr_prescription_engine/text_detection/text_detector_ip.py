import os

import cv2
import numpy as np
import imutils
from pythonRLSA import rlsa

def thresh_for_detect_address(gray_img, adaptive_value=3):
    """[summary]
    
    Arguments:
        gray_img {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    height = gray_img.shape[0]
    kernel_size = height // 2 + 1 - (height // 2) % 2
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                     kernel_size, adaptive_value)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    RADIUS = otsu.shape[0] // 10
    kernel = np.ones((2 * RADIUS, 2 * RADIUS), dtype='uint8')
    dilated = cv2.dilate(otsu, kernel)
    return cv2.bitwise_and(adaptive, dilated)

def remove_table(thresh_image, debug_dir=None, debug=False):
    # if debug:
    #     debug_dir = os.path.join(debug_dir, '0_REMOVE_TABLE')
    #     os.makedirs(debug_dir, exist_ok=True)
    height, width = thresh_image.shape
    vertical_thresh_image = cv2.erode(thresh_image.copy(), kernel=np.ones((3, 1)), anchor=(0, 1),
                                      iterations=height // 4)
    vertical_thresh_image = cv2.dilate(vertical_thresh_image, kernel=np.ones((3, 1)), anchor=(0, 1),
                                       iterations=height // 4)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(vertical_thresh_image, 4, cv2.CV_32S)
    # if debug:
    #     cv2.imwrite(os.path.join(debug_dir, '0_vertical_thresh_image.jpg'), vertical_thresh_image)
    thresh_image_vertical = thresh_image.copy()
    boxes = stats[1:]
    for box in boxes:
        x, y, w, h, _ = box
        if w < min(height * 0.1, 20) and h == height and h / w > 7:
            if x < width * 0.1:
                thresh_image_vertical[y:y + h, x:x+w] = 0
            elif x > width * 0.9:
                thresh_image_vertical[y:y + h, x:x+w] = 0

    horizontal_thresh_image = cv2.erode(thresh_image.copy(), kernel=np.ones((1, 3)), anchor=(1, 0),
                                        iterations=width // 3)
    horizontal_thresh_image = cv2.dilate(horizontal_thresh_image, kernel=np.ones((1, 3)), anchor=(1, 0),
                                         iterations=width // 3)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(horizontal_thresh_image, 4, cv2.CV_32S)
    # if debug:
    #     cv2.imwrite(os.path.join(debug_dir, '1_horizontal_thresh_image.jpg'), horizontal_thresh_image)

    thresh_image_horizontal = thresh_image.copy()
    boxes = stats[1:]
    for box in boxes:
        x, y, w, h, _ = box
        if h < height * 0.2 and w > width * 0.9 and w / h > 7:
            thresh_image_horizontal[y:y + h, x: x + w] = 0
        elif h > height * 0.8 and w > width * 0.9 and w / h > 7:
            thresh_image_horizontal[y:y + h, x: x + w] = 0

    thresh_image = cv2.bitwise_and(thresh_image, thresh_image_horizontal)
    thresh_image = cv2.bitwise_and(thresh_image, thresh_image_vertical)
    # if debug:
    #     cv2.imwrite(os.path.join(debug_dir, '2_thresh_image.jpg'), thresh_image)
    return thresh_image

def order_points(points):
    """
    Sort 4 points of box
    Arguments:
        points {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    # sort the points based on their x-coordinates
    x_sorted = points[np.argsort(points[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (top_left, bottom_left) = left_most

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    # D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    # (br, tr) = rightMost[np.argsort(D)[::-1], :]

    right_most = right_most[np.argsort(right_most[:, 1]), :]
    (top_right, bottom_right) = right_most

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

def sort_bounding_box(bounding_boxes):
    add_space_dict = {}
    if len(bounding_boxes) == 1:
        add_space_dict[0] = '\n'
        return bounding_boxes, add_space_dict
    final_sorted_bounding_boxes = []
    line_index = 0
    count = 0
    while count < 20:
        if len(bounding_boxes) == 0:
            break
        else:
            sorted_bounding_boxes = []
            sorted_y_bounding_boxes = [bounding_boxes[i] for i in np.argsort(
                np.array([np.average([box[2, 1], box[0, 1], box[3, 1], box[1, 1]]) for box in bounding_boxes]))]
            tl_y_min = sorted_y_bounding_boxes[0][0][1]
            tl_y_max = sorted_y_bounding_boxes[0][3][1]
            pop_index = []
            for i, box in enumerate(bounding_boxes):
                box_high = np.median(
                    [box[2, 1] + box[0, 1], box[3, 1] + box[1, 1]]) // 2
                if tl_y_min - 2 < box_high < tl_y_max + 2:
                    sorted_bounding_boxes.append(box)
                    pop_index.append(i)
            for i in sorted(pop_index, reverse=True):
                del bounding_boxes[i]
            sorted_bounding_boxes = [sorted_bounding_boxes[idx] for idx in np.array(
                [box[0, 0] for box in sorted_bounding_boxes]).argsort()]
            for i in range(len(sorted_bounding_boxes)):
                add_space_dict[i + line_index] = ' '
            final_sorted_bounding_boxes += sorted_bounding_boxes
            line_index += len(sorted_bounding_boxes)
            add_space_dict[line_index - 1] = '\n'
            count += 1

    return np.array(final_sorted_bounding_boxes), add_space_dict

def split_image(gray_image, thresh_img, rlsa_length=120, erode_dilate=[2, 2, 1], kernel_size=(3, 3), debug=False,
                debug_dir='logs', debug_name='CROP_AND_SPLIT'):
    """
    Separate image
    Arguments:
        thresh_img {[type]} -- [description]

    Keyword Arguments:
        thresh_img_origin {[type]} -- [description] (default: {None})
        debug {bool} -- [description] (default: {False})
        debug_dir {str} -- [description] (default: {'logs'})
        max_lines {[type]} -- [description] (default: {None})
        field_name {str} -- [description] (default: {''})
        rlsa_length {int} -- [description] (default: {120})
        erode_dilate {list} -- [description] (default: {[3, 3, 1]})
        kernel_size {tuple} -- [description] (default: {(3, 3)})

    Returns:
        [type] -- [description]
    """
    # if debug:
    #     log_dir = os.path.join(debug_dir, debug_name)
    #     os.makedirs(log_dir, exist_ok=True)
    original_img = thresh_img.copy()
    _, bin_img = cv2.threshold(thresh_img, 127, 255, cv2.THRESH_BINARY_INV)
    image_height, image_width = thresh_img.shape

    # connect all chars in the same line
    image_rlsa_horizontal = rlsa.rlsa(bin_img, True, False, rlsa_length)
    _, image_rlsa = cv2.threshold(image_rlsa_horizontal, 250, 255, cv2.THRESH_BINARY_INV)
    # if debug:
    #     cv2.imwrite(os.path.join(log_dir, '2_image_rlsa.jpg'), image_rlsa)
    #     cv2.imwrite(os.path.join(log_dir, '0_gray_image.jpg'), gray_image)
    #     cv2.imwrite(os.path.join(log_dir, '1_thresh_img.jpg'), thresh_img)

    connects = cv2.connectedComponentsWithStats(image_rlsa, labels=255, connectivity=8)
    boxes = connects[2][1:]

    if not len(boxes) == 1:
        # remove small noise
        image_rlsa = cv2.erode(image_rlsa, np.ones(kernel_size), iterations=erode_dilate[0])
        # if debug:
        #     cv2.imwrite(os.path.join(log_dir, '3_image_rlsa_erode.jpg'), image_rlsa)
        image_rlsa = cv2.dilate(image_rlsa, np.ones(kernel_size), iterations=erode_dilate[1])
        # if debug:
        #     cv2.imwrite(os.path.join(log_dir, '4_image_rlsa_dilate.jpg'), image_rlsa)

    # find contours
    cnts = cv2.findContours(image_rlsa, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(cnts)

    if len(contours) == 0:
        return []

    # convert to bounding box (tl, tr, br, bl)
    bounding_boxes = [np.int32(order_points(
        cv2.boxPoints(cv2.minAreaRect(rect)))) for rect in contours]

    origin_sorted_bounding_boxes, add_space_dict = sort_bounding_box(bounding_boxes)

    # get each line of the thresh image and concatenate imgs
    final_thresh_images = []
    crop_gray_images = []
    origin_add_space_list = [add_space_dict[_] for _ in add_space_dict.keys()]

    add_space_list = []
    sorted_bounding_boxes = []
    small_boxes = []
    for i, box in enumerate(origin_sorted_bounding_boxes):
        width = np.average([box[1, 0] - box[0, 0], box[2, 0] - box[3, 0]])
        height = np.average([box[2, 1] - box[0, 1], box[3, 1] - box[1, 1]])
        if (height < 22 and image_width > 1.1 * image_height) or (
                width < 10 and image_width <= 1.1 * image_height) or width < 5 or height < 5:
            if origin_add_space_list[i] == '\n':
                if len(add_space_list) == 0:
                    add_space_list.append('\n')
                else:
                    add_space_list[-1] = '\n'
            # Save ignore box to readd later
            small_boxes.append(box)
        else:
            add_space_list.append(origin_add_space_list[i])
            sorted_bounding_boxes.append(box)

    # Concate near box for repair small stroke
    for box in small_boxes:
        box_x_min = int(np.average([box[0, 0], box[3, 0]]))
        box_y_min = int(np.average([box[0, 1], box[1, 1]]))
        box_width = int(np.average([box[1, 0] - box[0, 0], box[2, 0] - box[3, 0]]))
        box_height = int(np.average([box[2, 1] - box[0, 1], box[3, 1] - box[1, 1]]))
        # Loop list big stroke to find match case
        for index, tmp_box in enumerate(sorted_bounding_boxes):
            tmp_box_x_min = int(np.average([tmp_box[0, 0], tmp_box[3, 0]]))
            tmp_box_y_min = int(np.average([tmp_box[0, 1], tmp_box[1, 1]]))
            tmp_box_width = int(np.average([tmp_box[1, 0] - tmp_box[0, 0], tmp_box[2, 0] - tmp_box[3, 0]]))
            tmp_box_height = int(np.average([tmp_box[2, 1] - tmp_box[0, 1], tmp_box[3, 1] - tmp_box[1, 1]]))
            if box_y_min < tmp_box_y_min and tmp_box_y_min - box_y_min - box_height < 1.2 * box_height and box_y_min + box_height in range(
                    int(image_height * 0.2), int(image_height * 0.8)):
                # Found near box that have reasonable width with small box on top
                if box_width < tmp_box_width and abs(box_x_min - tmp_box_x_min) < 1.1 * tmp_box_width and abs(
                        box_x_min + box_width - tmp_box_x_min - tmp_box_width) < 1.1 * tmp_box_width:
                    sorted_bounding_boxes[index] = np.array([
                        [min(box_x_min, tmp_box_x_min), min(box_y_min, tmp_box_y_min)],
                        [max(box_x_min + box_width, tmp_box_x_min + tmp_box_width), min(box_y_min, tmp_box_y_min)],
                        [max(box_x_min + box_width, tmp_box_x_min + tmp_box_width),
                         max(box_y_min + box_height, tmp_box_y_min + tmp_box_height)],
                        [min(box_x_min, tmp_box_x_min), max(box_y_min + box_height, tmp_box_y_min + tmp_box_height)]
                    ])
            elif box_y_min > tmp_box_y_min and box_y_min - tmp_box_y_min - tmp_box_height < 1.2 * box_height and box_y_min + box_height in range(
                    int(image_height * 0.2), int(image_height * 0.8)):
                # Found near box that have reasonable width with small box on bottom
                if box_width < tmp_box_width and abs(box_x_min - tmp_box_x_min) < 1.1 * tmp_box_width and abs(
                        box_x_min + box_width - tmp_box_x_min - tmp_box_width) < 1.1 * tmp_box_width:
                    sorted_bounding_boxes[index] = np.array([
                        [min(box_x_min, tmp_box_x_min), min(box_y_min, tmp_box_y_min)],
                        [max(box_x_min + box_width, tmp_box_x_min + tmp_box_width), min(box_y_min, tmp_box_y_min)],
                        [max(box_x_min + box_width, tmp_box_x_min + tmp_box_width),
                         max(box_y_min + box_height, tmp_box_y_min + tmp_box_height)],
                        [min(box_x_min, tmp_box_x_min), max(box_y_min + box_height, tmp_box_y_min + tmp_box_height)]
                    ])
    text_boxes = []
    for i, box in enumerate(sorted_bounding_boxes):
        contour_box_img = image_rlsa
        contour_box_img[:, :] = 0
        cv2.fillPoly(contour_box_img, [box], 255)
        kernel = np.ones((3, 3))
        contour_box_img = cv2.dilate(contour_box_img, kernel, iterations=erode_dilate[2])
        contour_box_img = crop_add_boder(contour_box_img, add_left_right_factor=1)
        final_thresh_image = cv2.bitwise_and(original_img, contour_box_img)
        result = np.where(final_thresh_image > 0)
        y_min = min(result[0])  # min in height
        y_max = max(result[0])  # max in height
        x_min = min(result[1])  # max in height
        x_max = max(result[1])  # max in height
        text_boxes.append([x_min, y_min, x_max, y_max])
        # text_boxes.append([x_min, y_min, x_max - x_min, y_max - y_min, 0])
    return text_boxes

def crop_add_boder(img, add_boder_value=None, add_left_right_factor=None,
                   add_top_down_factor=None, add_all=False,
                   get_box=False):
    """
    Crop text area then add border
    Arguments:
        img {[type]} -- [description]

    Keyword Arguments:
        add_boder_value {[type]} -- [description] (default: {None})
        add_left_right_factor {[type]} -- [description] (default: {None})
        add_top_down_factor {[type]} -- [description] (default: {None})
        add_all {bool} -- [description] (default: {False})
        get_box {bool} -- [description] (default: {False})

    Returns:
        [type] -- [description]
    """
    try:
        height, weight = img.shape
        result = np.where(img > 0)
        x_min = min(result[1])  # min in width
        x_max = max(result[1])  # max in width
        y_min = min(result[0])  # min in height
        y_max = max(result[0])  # max in height
    except Exception:
        return img

    add_lr = (y_max - y_min) // 10
    add_td = (y_max - y_min) // 15
    if get_box:
        return [y_min, y_max, x_min, x_max]
    # to add boder to image
    if add_boder_value is not None:
        crop_img = img[y_min: y_max + 1, x_min: x_max + 1]
        crop_img = cv2.copyMakeBorder(crop_img, add_boder_value,
                                      add_boder_value, add_boder_value,
                                      add_boder_value,
                                      cv2.BORDER_CONSTANT, (0, 0, 0))
        return crop_img
    # to add value in left_right side
    if add_left_right_factor is not None:
        add_lr = int(add_lr * add_left_right_factor)
        img[y_min:y_max + 1, max(x_min - add_lr, 0): x_min] = 255
        img[y_min:y_max + 1, x_max: min(x_max + add_lr, weight)] = 255
        # img[y_min:y_max + 1, max(x_min - add_lr, 0): min(x_max + add_lr + 1, w)] = 255
    # to add value in top_down side
    if add_top_down_factor is not None:
        add_td = int(add_td * add_top_down_factor)
        img[max(y_min - add_td, 0):min(y_max + add_td + 1, height),
        x_min: x_max + 1] = 255
    # to add value in all 4 sides
    if add_all:
        img[max(y_min - add_td, 0):min(y_max + add_td + 1, height),
        max(x_min - add_lr, 0): min(x_max + add_lr + 1, weight)] = 255
    return img

def ip_text_detect(gray_image, debug=False, debug_dir=None):
    # Using cv2 for detect line
    # if debug:
    #     debug_dir = os.path.join(debug_dir, '1_IMAGEPROCESSING_BOXES_PROCESS')
    #     os.makedirs(debug_dir, exist_ok=True)

    thresh_image = thresh_for_detect_address(gray_image, adaptive_value=20)
    height, width = gray_image.shape

    # Detect white image
    if np.sum(thresh_for_detect_address(gray_image, adaptive_value=50)) == 0 and np.sum(
            thresh_for_detect_address(gray_image, adaptive_value=30)) < 0.99 * height * width:
        return [], gray_image, None
    if np.sum(thresh_image) == 0:
        return [], gray_image, None

    thresh_image = thresh_for_detect_address(gray_image, adaptive_value=3)

    thresh_image = remove_table(thresh_image, debug_dir=debug_dir, debug=debug)

    text_boxes = split_image(gray_image.copy(),
                            thresh_image.copy(),
                            rlsa_length=int(height * 8),
                            debug=debug, debug_dir=debug_dir,
                            debug_name='2_IP_SPLIT')
    log_image_ip = None
    return text_boxes, thresh_image, log_image_ip