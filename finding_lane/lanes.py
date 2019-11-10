import cv2
import numpy as np

import matplotlib.pyplot as plt

# Install cv2 with anaconda:
# conda install -c conda-forge opencv=4.1.0

# Get edges using canny
def canny(image):
    # Convert the image to grey scale
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Reduce noise
    # 0 means the deviation is calculated using kernel size
    blur = cv2.GaussianBlur(grey, (5, 5), 0)

    # Edge detection
    # Gradients higher than 150 is picked and lower than 50 is dropped
    # Pixels with gradient between 50 and 150 are only picked if connected to an already picked pixel.
    canny = cv2.Canny(blur, 50, 150)

    return canny


def region_of_interest(image):
    height = image.shape[0]
    triangle = np.array([(200, height), (1100, height), (550, 250)])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [triangle], 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            if len(line) == 4:
                x1, y1, x2, y2 = line
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return line_image


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_line = []
    right_line = []
    if left_fit:
        left_fit_average = np.average(left_fit, 0)
        left_line = make_coordinates(image, left_fit_average)
    if right_fit:
        right_fit_average = np.average(right_fit, 0)
        right_line = make_coordinates(image, right_fit_average)

    return np.array([left_line, right_line])


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def main():
    cap = cv2.VideoCapture("test2.mp4")
    while cap.isOpened():
        _, frame = cap.read()

        if frame is None:
            break

        # Get canny image
        canny_image = canny(frame)

        # Masked image
        masked_image = region_of_interest(canny_image)

        lines = cv2.HoughLinesP(
            masked_image,
            2,
            np.pi / 180,
            100,
            np.array([]),
            minLineLength=40,
            maxLineGap=5,
        )

        averaged_lines = average_slope_intercept(frame, lines)

        line_image = display_lines(frame, averaged_lines)

        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
        cv2.imshow("result", combo_image)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
