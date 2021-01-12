import operator

import cv2
import numpy as np
import math
import sudoku_solution
from scipy import ndimage
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import model_from_json
import copy
import pickle

pickle_in = open("model_trained.p","rb")
model = pickle.load(pickle_in)

def write_solution_on_image(image, grid, user_grid):
    '''write solved sudoku grid on image'''
    print("user_gird in write solution on image", user_grid)
    size = 9
    width = image.shape[1] // 9
    height = image.shape[0] // 9
    for i in range(size):
        for j in range(size):
            if (user_grid[i][j] != 0):
                continue
            text = str(grid[i][j])
            off_set_x = width // 15
            off_set_y = height // 15
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_height, text_width), baseline = cv2.getTextSize(text, font, fontScale=1,thickness=3)  # Calculates the width and height of a text string.
            marginX = math.floor(width / 7)
            marginY = math.floor(height / 7)

            font_scale = 0.6 * (min(width, height) / max(text_height, text_width))
            text_height *= font_scale
            text_width *= font_scale
            bottom_left_corner_x = width * j + math.floor((width - text_width) / 2) + off_set_x
            bottom_left_corner_y = height * (i + 1) - math.floor((height - text_height) / 2) + off_set_y
            # draw a string text on any image
            image = cv2.putText(image, text, (bottom_left_corner_x, bottom_left_corner_y), font, font_scale,
                                (255, 0, 0), thickness=3, lineType=cv2.LINE_AA)
    return image




def largest_connected_component(image):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = np.array(image.astype('uint8'))
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, None, None, None, 8)
    sizes = stats[:, -1]

    if (len(sizes) <= 1):
        blank_image = np.zeros(image.shape)
        blank_image.fill(255)
        return blank_image
    max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1])
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    # img2.fill(255)
    img2[output == max_label] = 255
    return img2


def distance_between(pt1, pt2):
    """returns the scaler distance between two points"""
    a = pt2[0] - pt1[0]
    b = pt2[0] - pt1[0]
    return np.sqrt((a ** 2) + (b ** 2))


def check_for_square(corners):
    # using the criteria---> square has equal sides

    # 3d array dim is no of 2d, row, col

    # points of shape detected
    A = corners[0, 1, 2]
    B = corners[1, 1, 2]
    C = corners[2, 1, 2]
    D = corners[3, 1, 2]

    AB = distance_between(A, B)
    BC = distance_between(B, C)
    CD = distance_between(C, D)
    DA = distance_between(D, A)

    if (AB == BC == CD == DA):
        return True
    else:
        return False


def rearrange_clockwise(polygon):
    # use of operator.itemgetter with max and min allows us to get index of point
    # Each point is an array of 0 coordinate, hence the [0] getter, then [0] or [0] used to get x and y respectively.

    # Bottom-right point has the largest (x + y) value
    # Top-left has point smallest (x + y) value
    # Bottom-left point has smallest (x - y) value
    # Top-right point has largest (x - y) value
    print(polygon)
    bottom_right = 0
    top_left = 0
    top_right = 0
    bottom_left = 0
    # bottom right
    max = 0
    count = 0
    for j in range(4):
        for i in polygon[j][0]:
            count = count + i
        if count > max:
            max = count
            bottom_right = j
        count = 0

    # top_lefttt
    min = 0
    sum = 0
    for i in polygon[0][0]:
        sum = sum + i
    min = sum
    top_left = 0
    for j in range(4):
        for i in polygon[j][0]:
            count = count + i
        if count < min:
            min = count
            top_left = j
        count = 0

    # top_right
    top_right = 0
    max = 0
    for i in range(4):
        diff = polygon[i, 0][0]
        diff = diff - polygon[i, 0][1]
        print(diff)
        if diff > max:
            max = diff
            top_right = i

    # bottom_left
    bottom_left = 0
    min = polygon[0, 0][0] - polygon[0, 0][1]
    for i in range(4):
        diff = polygon[i, 0][0]
        diff = diff - polygon[i, 0][1]
        if diff < min:
            min = diff
            bottom_left = i

    coordinates = [polygon[top_left][0], polygon[bottom_left][0], polygon[bottom_right][0], polygon[top_right][0]]
    return coordinates


def get_best_shift(image):
    centre_x, centre_y = ndimage.measurements.center_of_mass(image)  # gives centre of image
    rows, cols = image.shape
    shiftx = np.round(cols / 2.0 - centre_x).astype(int)
    shifty = np.round(rows / 2.0 - centre_y).astype(int)
    return shiftx, shifty


def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))  # helps in transformation of image take input 2*3 matrix
    return shifted
counter = 0
def The_Main(counte):


    ar = 0
    count = 0
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    video.set(3, 1280)
    video.set(4, 720)

    while True:
        # capture frame by frame
        ret, frame = video.read()

        k = 0
        # our functions on the frame come here

        # preprocessing image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # gray scale
        gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 3)  # removes noise from image
        adaptive_threshold = cv2.adaptiveThreshold(gaussian_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY_INV, 11,
                                                   2)  # threshold set out for background and foreground pixels

        # getting the largest square
        # contours is numpy array of all continous point obtained in picture(contour)...the boundaries of a shape with the same intensity.
        # RETR_LIST ---> it creates the hieracrchy of all the contours in image,i.e who is outerone,innerone etc
        contours, hierarchy = cv2.findContours(adaptive_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # max is variable for maximum area and maxc for maximum contours
        maxc = [0]
        max = 0
        # loop to find the largest contour in given frame
        for i in range(len(contours)):
            # finding the perimeter of contour and contour approximation
            perimeter = cv2.arcLength(contours[i], True)  # true if curve is closed...perimeter means length of arc
            epsilon = 0.03 * perimeter
            approx = cv2.approxPolyDP(contours[i], epsilon, True)
            if cv2.contourArea(contours[i]) > 100000 and cv2.contourArea(contours[i]) > max and len(approx) == 4:
                # checking maximum contours

                max = cv2.contourArea(contours[i])
                maxc = approx
        # if contour have four corners then saving that frame and drawing contours
        if len(maxc) == 4:
            count = count + 1
        else:
            count = 0
        if len(maxc) == 4:
            cv2.drawContours(frame, [maxc], -1, (255, 0, 2), 3)
            cv2.drawContours(frame, maxc, -1, (0, 255), 8)
        # displaying contour edges and corner
        cv2.imshow('all', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if count == 4:
            cv2.imwrite("frame.jpg", frame)
            k = 1
            if k == 1:
                ar = maxc.copy()
                # check_for_square(ar)
                # checking if maxc is approx square

                (x, y) = adaptive_threshold.shape
                mask = np.zeros((x, y, 3), np.uint8)
                mask = cv2.drawContours(mask, [ar], -1, (255, 255, 255), -1)
                mask = cv2.drawContours(mask, ar, -1, (0, 255), 2)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))  # kernel for ellipitcal shape
                close = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)  # closes the pores in foreground
                div = np.float32(
                    frame) / close  # the closed and gray images are divided to get a narrow histogram which when normalized increases the brightness and contrast of image
                res = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))
                masked = cv2.bitwise_and(mask, res)
                masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

                ar = rearrange_clockwise(ar)
                dest = np.array(ar, np.float32)  # source
                nw = np.array([[0, 0], [0, 450], [450, 450], [450, 0]], np.float32)  # destinaton
                M = cv2.getPerspectiveTransform(dest, nw)
                output = cv2.warpPerspective(res, M, (450, 450))
                output = cv2.GaussianBlur(output, (3, 3), 0)
                # output = cv2.adaptiveThreshold(output,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
                output = cv2.rectangle(output, (0, 0), (450, 450), 0, 1)
                cv2.imshow("output", output)

                # extracting digits from grid
                size = 9
                grid = []
                for i in range(size):
                    row = []
                    for j in range(size):
                        row.append(0)
                    grid.append(row)

                height = output.shape[0] // 9
                width = output.shape[1] // 9

                offset_width = math.floor(width / 10)  # offset is used to get rid of boundaries
                offset_height = math.floor(height / 10)

                # divide the sudoku board into 9*9 square boxes
                # square containing numbers will be stored in crop_image
                max = 0
                max2 = 0
                for i in range(size):
                    for j in range(size):

                        # crop image with offset
                        # offset :  VISIBLE content & padding + border + scrollbar
                        #cropping the image by image matrix image[height,width]
                        crop_image = output[height * i + offset_height:height * (i + 1) - offset_height,
                                     width * j + offset_width:width * (j + 1) - offset_width]

                        # cropping images more
                        crop_image = crop_image[2:38,2:38]
                        #cv2.imshow("crop_image", crop_image)
                        #cv2.waitKey(0)

                        crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
                        crop_image = cv2.GaussianBlur(crop_image, (3, 3), 0)
                        _, crop_image = cv2.threshold(crop_image, 210, 255, cv2.THRESH_BINARY)




                        # has too little black pixels
                        # digit_pic_size**2 -> area of image of digit.It is in square
                        digit_pic_size = 28
                        crop_image = cv2.resize(crop_image,(digit_pic_size, digit_pic_size))


                        if crop_image.sum() >= digit_pic_size ** 2 * 255 - 255:
                            #print("digit size")
                            grid[i][j] = 0
                            continue  # move on if we have a white cell
                        #print("out")

                        # criteria 2 for detecting white cell
                        # huge white area in centre
                        centre_width = crop_image.shape[1] // 2  # column
                        centre_height = crop_image.shape[0] // 2  # row

                        x_start = centre_height // 2
                        x_end = centre_height // 2 + centre_height
                        y_start = centre_width // 2
                        y_end = centre_width // 2 + centre_width
                        centre_region = crop_image[x_start:x_end, y_start:y_end]

                        if centre_region.sum() >= centre_width * centre_height * 255 - 255:
                            print("in in secong if")
                            grid[i][j] = 0
                            continue  # move on if we have white cell
                        print("out from 2 if")
                        # now we dont have any white cell
                        #cv2.imshow("crop_image", crop_image)
                        #cv2.waitKey(0)

                        # centralize the image according to centre of mass
                        crop_image = cv2.bitwise_not(crop_image)
                        shift_x, shift_y = get_best_shift(crop_image)
                        shifted = shift(crop_image, shift_x, shift_y)
                        crop_image = shifted
                        crop_image = cv2.bitwise_not(crop_image)

                        #cv2.imshow("crop_image", crop_image)
                        #cv2.imwrite("crop()_image{0}.png".format(i), crop_image)
                        #cv2.waitKey(0)

                        _, crop_thresh = cv2.threshold(crop_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                        #cv2.imshow("crop_image", crop_thresh)
                        #cv2.imwrite("crop()_image{0}.png".format(i), crop_image)
                        #cv2.waitKey(0)

                        # recognizing digits

                        prediction = model.predict([crop_thresh.reshape(1, 28, 28, 1)])
                        grid[i][j] = np.argmax(prediction[0]) + 1
                    user_grid = copy.deepcopy(grid)
                    #print("grid", grid)
                    #print("user_grid", user_grid)

                    #################################################
                #Solving the sudoku
                sudoku_solved = sudoku_solution.search(sudoku_solution.parse_grid(grid))
                print(sudoku_solved)
                if(sudoku_solved == False):
                    print("NOISE ATTAINED>>>>>RESTART")
                    try:
                        The_Main(counter+1)
                    except:
                        if counter == 5:
                            print("This cannnot be solved")
                            exit()
                else:

                    sudoku_solved = list(sudoku_solved.values())
                    sudoku_solved = np.asarray(sudoku_solved).reshape(9, 9)
                    original_warp = write_solution_on_image(output, sudoku_solved, user_grid)
                    # cv2.imshow("write sol on image", original_warp)
                    # cv2.waitKey(0)
                    old_sudoku = copy.deepcopy(grid)
                    dest = np.array(ar, np.float32)
                    nw = np.array([[0, 0], [0, 450], [450, 450], [450, 0]], np.float32)
                    M = cv2.getPerspectiveTransform(dest, nw)
                    #apply inverse perspective transform and paste the solutions on top of the original image
                    result_sudoku = cv2.warpPerspective(original_warp, M, (frame.shape[1], frame.shape[0]),
                                                        flags=cv2.WARP_INVERSE_MAP)
                    print(np.asarray(result_sudoku).shape)
                    result = np.where(result_sudoku.sum(axis=-1, keepdims=True) != 0, result_sudoku, frame)
                    print("reached here")
                    cv2.imshow("final image", result)
                    cv2.waitKey(0)
    video.release()
    cv2.destroyAllWindows()

The_Main(1)