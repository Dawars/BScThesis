import numpy as np
import cv2


def normalize_image(image):
    """
    Normalizes input image
    :param image: image to be normalized
    :return: numpy array of floats with values between [-1, 1]
    """
    return (image.astype(np.float) / 255) * 2.0 - 1.0


def denormalize_image(image):
    """
        Reverses normalization of input image
        :param image: image to be denormalized
        :return: numpy array of ints with values between [0, 255]
    """
    return ((image + 1) / 2 * 255).astype(np.int)


def flip_image_horizontal(image):
    """
        Flips the image horizontally
        :param image: image to be flipped
        :return: flipped image
    """
    return cv2.flip(image, 0)


def flip_image_vertical(image):
    """
        Flips the image vertically
        :param image: image to be flipped
        :return: flipped image
    """
    return cv2.flip(image, 1)


def rotate_image_180(image):
    """
        Rotate image by 180 deg
        :param image: image to be roatated
        :return: rotated image
    """
    return cv2.flip(image, -1)


def crop_image(image, joints, p1, p2):
    """
    Crop image
    :param image: image to crop
    :param joints: joints
    :param p1: top left corner
    :param p2: bottom right corner
    :return: cropped image
    """
    joints[:, :2] -= p1

    return image[p1[1]:p2[1], p1[0]:p2[0]], joints


def crop_img(image, joints, padding=50):
    """
    crop image to rectengular, resize to size
    :param image: image to crop
    :param joints: list of joint positions in pixels
    :param padding: padding outside of joint positions
    :param size: final image size
    :return: img in bgr format and rescaled joint locations
    """
    height, width = image.shape[:2]

    joints = np.array(joints)
    y_min = int(max(0, joints[:, 1].min() - padding))
    y_max = int(min(height, joints[:, 1].max() + padding))
    bb_height = y_max - y_min
    x_middle = (joints[:, 0].min() + joints[:, 0].max()) / 2  # middle of hand in x direction

    x_min = int(x_middle - bb_height / 2)
    x_max = int(x_middle + bb_height / 2)

    if x_min < 0:
        x_max += 0 - x_min
        x_min = 0
    if x_max > width:
        x_min += width - x_max
        x_max = width

    # substract corner coordinates
    joints[:, :2] -= (x_min, y_min)
    return image[y_min:y_max, x_min:x_max], joints


def draw_joints(image, kps, color=(255, 0, 0)):
    """
    Draw joint locations on image, the more uncertain a joint position, the darker it is
    :param image: in HWC format [0, 255] range
    :param kps: list of joint locations in pixels
    :param color: color of joints
    :return:
    """
    src_image = image.copy()
    for pt in kps:
        if len(pt) > 2:
            col = (pt[2] * color[0], pt[2] * color[1], pt[2] * color[2])
        else:
            col = color

        size = image.shape[0] // 100
        cv2.circle(src_image, (int(pt[0]), int(pt[1])), size, col, -1)

    return src_image


def hwc2chw(image):
    """
    Converts image from HWC to CHW format (C: channel, H: height, W: width)
    :param image:
    :return:
    """
    return image.transpose((2, 0, 1))


def chw2hwc(image):
    """
    Converts image from CHW to HWC format (C: channel, H: height, W: width)
    :param image:
    :return:
    """
    return image.transpose((1, 2, 0))


def get_bounding_box(joints):
    """
    Calculates the bounding box of the joints
    :param joints:
    :return: (p1, p2)
    """
    xmin = int(joints[:, 0].min() - 100)
    xmax = int(joints[:, 0].max() + 100)
    ymin = int(joints[:, 1].min() - 100)
    ymax = int(joints[:, 1].max() + 100)
    return (xmin, ymin), (xmax, ymax)


def get_bounding_rect(image, joints):
    """
    Calculates the bounding rectangle of the joints
    :param image:
    :param joints:
    :return: (p1, p2)
    """
    im_h, im_w = image.shape[:2]

    xmin = int(joints[:, 0].min() - 50)
    xmax = int(joints[:, 0].max() + 50)
    ymin = int(joints[:, 1].min() - 50)
    ymax = int(joints[:, 1].max() + 50)

    w, h = xmax - xmin, ymax - ymin
    half_side = min(im_h, max(w, h)) // 2

    midx = (xmin + xmax) // 2
    midy = im_h // 2

    return (midx - half_side, midy - half_side), (midx + half_side, midy + half_side)


def resize_image(image, joints, size=(224, 224)):
    """
    Resize image and joint locations together
    :param image:
    :param joints:
    :param size:
    :return:
    """
    shape_ = (image.shape[1], image.shape[0])
    joints[:, :2] *= np.asarray(size) / shape_
    size_t = (size[1], size[0])
    return cv2.resize(image, size_t, interpolation=cv2.INTER_CUBIC), joints


def concat_images(images):
    """
    Concatenate images horizontally
    :param images: list of images in HWC
    :return: concatenated image
    """
    h, w = images[0].shape[:2]
    d = np.zeros((h, w * len(images), 3), dtype=np.int)
    for i, img in enumerate(images):
        d[:, i * w:(i + 1) * w, :] = img[:, :, :]

    return d
