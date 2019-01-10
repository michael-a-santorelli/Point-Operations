'''
This file contains 4 digital image processing point operation functions.
These 4 point operations include:
    - Brightness Adjustment,
    - Image Negative,
    - Contrast Stretching, and
    - Piece-wise Transform
Brightness Adjustment is used by calling brightnessAdjust(...),
Image Negative is used by calling imageNegative(...),
Contrast Stretching is used by calling constrastStretching(...),
Piece-wise Transform is used by calling piecewiseTransform(...).
Each of these 4 functions have their own documentation,
for further help call help(function_name_here)
'''
import numpy
import cv2


# Brightness Adjustment function
def brightnessAdjust(image: str, colorInfo: str, adjustSlider: int):
    '''
    This function takes 3 parameters:
        - The first parameter is a string which represents the file name of the image to be processed.
        - The second parameter is a string which represents the type of output of the processed image;
          this is either "gray" for a grayscaled output, or any other string for a colored output.
        - The third parameter is an integer which represents the amount of brightness to be added or 
          subtracted from the image.
    This function adds (or subtracts) multiples of 10% of brightness to the image depending on the "adjustSlider"
    parameter input.
    Example calls:
        brightnessAdjust("test.jpg", "gray", 3) - outputs a grayscaled image with 30% brighness added
        brighnessAdjust("test.jpg", "gray", -1) - outputs a grayscaled image with 10% brighness subtracted
        brightnessAdjust("test.jpg", "color", -2) - outputs a color image with 20% brighness subtracted
        brightnessAdjust("test.jpg", "color", 4) - outputs a color image with 40% brighness added
    '''
    # Make look-up table for manipulating data
    LUT = {}
    for i in range(256):
        LUT[i] = i + 0.1*i*adjustSlider

    # Load image
    img = cv2.imread(image)
    # If user wants grayscale output, reload as grayscale image
    if(colorInfo == "gray"):
        img = cv2.imread(image, 0)
    # Determine image size; used for creating output image
    imgSize = img.shape

    # If image number of dimensions is 2 perform grayscale operations
    if(img.ndim == 2):
        # Create output matrix of input image size
        output = numpy.zeros([imgSize[0], imgSize[1]], dtype=numpy.float32)
        # Apply point operation for brightness adjustment
        for i in range(256):
            output[numpy.where(img == i)] = LUT[i]

        # Determine if operation has caused uint8 overflow;
        # if so, set overflown values to 255 to eliminate overflow
        if(numpy.amax(output) > 255.0):
            output[numpy.where(output > 255.0)] = 255.0

    # If image dimension is not 2 it will be 3; perform color manipulations
    else:
        # Create output matrices for blue, green, and red channels
        # of input matrix size
        outputB = numpy.zeros([imgSize[0], imgSize[1]], dtype=numpy.float32)
        outputG = numpy.zeros([imgSize[0], imgSize[1]], dtype=numpy.float32)
        outputR = numpy.zeros([imgSize[0], imgSize[1]], dtype=numpy.float32)
        # Create output matrix of input image size with 3 channels
        output = numpy.zeros([imgSize[0], imgSize[1], 3], dtype=numpy.float32)

        # Assign blue, green, and red channels to b, g, r, respectively
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]

        # Apply point operation for brightness adjustment
        for i in range(256):
                outputB[numpy.where(b == i)] = LUT[i]
                outputG[numpy.where(g == i)] = LUT[i]
                outputR[numpy.where(r == i)] = LUT[i]

        # Set blue transformations, green transformations, and
        # red transformations to channels 0, 1, and 2 of the output
        for i in range(imgSize[0]):
            for j in range(imgSize[1]):
                output[i, j, 0] = outputB[i, j]
                output[i, j, 1] = outputG[i, j]
                output[i, j, 2] = outputR[i, j]

        # Determine if operation has caused uint8 overflow;
        # if so, set overflown values to 255 to eliminate overflow
        if(numpy.amax(output) > 255.0):
            output[numpy.where(output > 255.0)] = 255.0

    # Convert output to uint8 type for displaying purposes
    output = numpy.array(output, dtype=numpy.uint8)

    # Display input and output image
    cv2.imshow("input", img)
    cv2.imshow("output", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Image Negative function
def imageNegative(image: str, colorInfo: str):
    '''
    This function takes 2 parameters:
        - The first parameter is a string which represents the file name of the image to be processed.
        - The second parameter is a string which represents the type of output of the processed image;
          this is either "gray" for a grayscaled output, or any other string for a colored output.
    This function subtracts each pixel value from 255 to obtain a "negative" of the image.
    Example calls:
        imageNegative("test.jpg", "gray") - outputs a grayscaled "negative" of the input image
        imageNegative("test.jpg", "color") - outputs a color "negative" of the input image
    '''
    # Make look-up table for manipulating data
    LUT = {}
    for i in range(256):
        LUT[i] = 255 - i

    # Load image
    img = cv2.imread(image)
    # If user wants grayscale output, reload as grayscale image
    if(colorInfo == "gray"):
        img = cv2.imread(image, 0)
    # Determine image size; used for creating output image
    imgSize = img.shape

    # If image number of dimensions is 2 perform grayscale operations
    if(img.ndim == 2):
        # Create output matrix of input image size
        output = numpy.zeros([imgSize[0], imgSize[1]], dtype=numpy.float32)
        # Apply point operation for image negative
        for i in range(256):
            output[numpy.where(img == i)] = LUT[i]

    # If image dimension is not 2 it will be 3; perform color manipulations
    else:
        # Create output matrices for blue, green, and red channels
        # of input matrix size
        outputB = numpy.zeros([imgSize[0], imgSize[1]], dtype=numpy.float32)
        outputG = numpy.zeros([imgSize[0], imgSize[1]], dtype=numpy.float32)
        outputR = numpy.zeros([imgSize[0], imgSize[1]], dtype=numpy.float32)
        # Create output matrix of input image size with 3 channels
        output = numpy.zeros([imgSize[0], imgSize[1], 3], dtype=numpy.float32)

        # Assign blue, green, and red channels to b, g, r, respectively
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]

        # Apply point operation for image negative
        for i in range(256):
                outputB[numpy.where(b == i)] = LUT[i]
                outputG[numpy.where(g == i)] = LUT[i]
                outputR[numpy.where(r == i)] = LUT[i]

        # Set blue transformations, green transformations, and
        # red transformations to channels 0, 1, and 2 of the output
        for i in range(imgSize[0]):
            for j in range(imgSize[1]):
                output[i, j, 0] = outputB[i, j]
                output[i, j, 1] = outputG[i, j]
                output[i, j, 2] = outputR[i, j]

    # Convert output to uint8 type for displaying purposes
    output = numpy.array(output, dtype=numpy.uint8)

    # Display input and output image
    cv2.imshow("input", img)
    cv2.imshow("output", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Contrast Stretching function
def contrastStretching(image: str, colorInfo: str):
    '''
    This function takes 2 parameters:
        - The first parameter is a string which represents the file name of the image to be processed.
        - The second parameter is a string which represents the type of output of the processed image;
          this is either "gray" for a grayscaled output, or any other string for a colored output.
    This function attemps to add contrast by stretching the dynamic range of pixel values from 0 to 255.
    If the minimum is already 0, and the maximum is already 255, this function will have no effect on the image.
    Example calls:
        contrastStretching("test.jpg", "gray") - outputs a grayscaled, contrast stretched image
        contrastStretching("test.jpg", "color") - outputs a color, contrast stretched image
    '''
    # Load image
    img = cv2.imread(image)
    # If user wants grayscale output, reload as grayscale image
    if(colorInfo == "gray"):
        img = cv2.imread(image, 0)
    # Determine image size; used for creating output image
    imgSize = img.shape

    # Make look-up table for manipulating data
    LUT = {}
    for i in range(256):
        LUT[i] = (255) * \
            ((i - numpy.amin(img)) / (numpy.amax(img) - numpy.amin(img)))

    # If image number of dimensions is 2 perform grayscale operations
    if(img.ndim == 2):
        # Create output matrix of input image size
        output = numpy.zeros([imgSize[0], imgSize[1]], dtype=numpy.float32)
        # Apply point operation for contrast streching
        for i in range(256):
            output[numpy.where(img == i)] = LUT[i]

    # If image dimension is not 2 it will be 3; perform color manipulations
    else:
        # Create output matrices for blue, green, and red channels
        # of input matrix size
        outputB = numpy.zeros([imgSize[0], imgSize[1]], dtype=numpy.float32)
        outputG = numpy.zeros([imgSize[0], imgSize[1]], dtype=numpy.float32)
        outputR = numpy.zeros([imgSize[0], imgSize[1]], dtype=numpy.float32)
        # Create output matrix of input image size with 3 channels
        output = numpy.zeros([imgSize[0], imgSize[1], 3], dtype=numpy.float32)

        # Assign blue, green, and red channels to b, g, r, respectively
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]

        # Apply point operation for contrast streching
        for i in range(256):
                outputB[numpy.where(b == i)] = LUT[i]
                outputG[numpy.where(g == i)] = LUT[i]
                outputR[numpy.where(r == i)] = LUT[i]

        # Set blue transformations, green transformations, and
        # red transformations to channels 0, 1, and 2 of the output
        for i in range(imgSize[0]):
            for j in range(imgSize[1]):
                output[i, j, 0] = outputB[i, j]
                output[i, j, 1] = outputG[i, j]
                output[i, j, 2] = outputR[i, j]

    # Convert output to uint8 type for displaying purposes
    output = numpy.array(output, dtype=numpy.uint8)

    # Display input and output image
    cv2.imshow("input", img)
    cv2.imshow("output", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Piece-wise Transform function
def piecewiseTransform(image: str, colorInfo: str, P1, P2):
    '''
    This function takes 4 parameters:
        - The first parameter is a string which represents the file name of the image to be processed.
        - The second parameter is a string which represents the type of output of the processed image;
          this is either "gray" for a grayscaled output, or any other string for a colored output.
        - The third parameter is a list which represents the first of two points used in creating a linear
          piece-wise transformation
        - The fourth parameter is a list which represents the second of two points used in creating a linear
          piece-wise transformation
    This function uses two points, P1 and P2, to create a linear piece-wise transformation. The lists, P1
    and P2, can be thought of as "x" and "y" coordinates.
    e.g.
        P1(x) = P1[0],
        P1(y) = P1[1]
    Similarly,
        P2(x) = P2[0],
        P2(y) = P2[1]
    This function can therefore be used to add contrast, or perform many different operations such as
    adding brightness, removing brightness, etc.
    Example calls:
        piecewiseTransform("test.jpg", "gray", [50, 5], [150, 225]) - outputs a grayscaled image processed using P1 and P2
        piecewiseTransform("test.jpg", "color", [50, 5], [150, 225]) - outputs a color image processed using P1 and P2
    '''
    # Load image
    img = cv2.imread(image)
    # If user wants grayscale output, reload as grayscale image
    if(colorInfo == "gray"):
        img = cv2.imread(image, 0)
    # Determine image size; used for creating output image
    imgSize = img.shape

    # Make look-up table for manipulating data
    LUT = {}
    for i in range(P1[0]):
        LUT[i] = i * P1[1]/P1[0]
    for i in range(P1[0], P2[0]):
        LUT[i] = P1[1] + ((P2[1] - P1[1]) / (P2[0] - P1[0]) * (i - P1[0]))
    for i in range(P2[0], 256):
        LUT[i] = P2[1] + ((255 - P2[1]) / (255 - P2[0]) * (i - P2[0]))

    # If image number of dimensions is 2 perform grayscale operations
    if(img.ndim == 2):
        # Create output matrix of input image size
        output = numpy.zeros([imgSize[0], imgSize[1]], dtype=numpy.float32)
        # Apply point operation for piece-wise transformation
        for i in range(256):
            output[numpy.where(img == i)] = LUT[i]

    # If image dimension is not 2 it will be 3; perform color manipulations
    else:
        # Create output matrices for blue, green, and red channels
        # of input matrix size
        outputB = numpy.zeros([imgSize[0], imgSize[1]], dtype=numpy.float32)
        outputG = numpy.zeros([imgSize[0], imgSize[1]], dtype=numpy.float32)
        outputR = numpy.zeros([imgSize[0], imgSize[1]], dtype=numpy.float32)
        # Create output matrix of input image size with 3 channels
        output = numpy.zeros([imgSize[0], imgSize[1], 3], dtype=numpy.float32)

        # Assign blue, green, and red channels to b, g, r, respectively
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]

        # Apply point operation for piece-wise transformation
        for i in range(256):
                outputB[numpy.where(b == i)] = LUT[i]
                outputG[numpy.where(g == i)] = LUT[i]
                outputR[numpy.where(r == i)] = LUT[i]

        # Set blue transformations, green transformations, and
        # red transformations to channels 0, 1, and 2 of the output
        for i in range(imgSize[0]):
            for j in range(imgSize[1]):
                output[i, j, 0] = outputB[i, j]
                output[i, j, 1] = outputG[i, j]
                output[i, j, 2] = outputR[i, j]

    # Convert output to uint8 type for displaying purposes
    output = numpy.array(output, dtype=numpy.uint8)

    # Display input and output image
    cv2.imshow("input", img)
    cv2.imshow("output", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
