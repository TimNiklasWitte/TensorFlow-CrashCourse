import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt

# for the color conversion see https://github.com/tensorflow/io/blob/v0.24.0/tensorflow_io/python/experimental/color_ops.py#L398-L459
# code from tensorflow io we're used, because tfio has no pip package for the jetson nano
def rgb_to_lab(input, illuminant="D65", observer="2", name=None):
    """
    Convert a RGB image to CIE LAB.
    Args:
    input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    illuminant : {"A", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10"}, optional
        The aperture angle of the observer.
    name: A name for the operation (optional).
    Returns:
    A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    input = tf.convert_to_tensor(input)
    assert input.dtype in (tf.float16, tf.float32, tf.float64)

    illuminants = {
        "A": {
            "2": (1.098466069456375, 1, 0.3558228003436005),
            "10": (1.111420406956693, 1, 0.3519978321919493),
        },
        "D50": {
            "2": (0.9642119944211994, 1, 0.8251882845188288),
            "10": (0.9672062750333777, 1, 0.8142801513128616),
        },
        "D55": {
            "2": (0.956797052643698, 1, 0.9214805860173273),
            "10": (0.9579665682254781, 1, 0.9092525159847462),
        },
        "D65": {
            "2": (0.95047, 1.0, 1.08883),
            "10": (0.94809667673716, 1, 1.0730513595166162),
        },
        "D75": {
            "2": (0.9497220898840717, 1, 1.226393520724154),
            "10": (0.9441713925645873, 1, 1.2064272211720228),
        },
        "E": {"2": (1.0, 1.0, 1.0), "10": (1.0, 1.0, 1.0)},
    }
    coords = tf.constant(illuminants[illuminant.upper()][observer], input.dtype)

    xyz = rgb_to_xyz(input)

    xyz = xyz / coords

    xyz = tf.where(
        tf.math.greater(xyz, 0.008856),
        tf.math.pow(xyz, 1.0 / 3.0),
        xyz * 7.787 + 16.0 / 116.0,
    )

    xyz = tf.unstack(xyz, axis=-1)
    x, y, z = xyz[0], xyz[1], xyz[2]

    # Vector scaling
    l = (y * 116.0) - 16.0
    a = (x - y) * 500.0
    b = (y - z) * 200.0

    return tf.stack([l, a, b], axis=-1)


def lab_to_rgb(input, illuminant="D65", observer="2", name=None):
    """
    Convert a CIE LAB image to RGB.
    Args:
    input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    illuminant : {"A", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10"}, optional
        The aperture angle of the observer.
    name: A name for the operation (optional).
    Returns:
    A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    input = tf.convert_to_tensor(input)
    assert input.dtype in (tf.float16, tf.float32, tf.float64)

    lab = input
    lab = tf.unstack(lab, axis=-1)
    l, a, b = lab[0], lab[1], lab[2]

    y = (l + 16.0) / 116.0
    x = (a / 500.0) + y
    z = y - (b / 200.0)

    z = tf.math.maximum(z, 0)

    xyz = tf.stack([x, y, z], axis=-1)

    xyz = tf.where(
        tf.math.greater(xyz, 0.2068966),
        tf.math.pow(xyz, 3.0),
        (xyz - 16.0 / 116.0) / 7.787,
    )

    illuminants = {
        "A": {
            "2": (1.098466069456375, 1, 0.3558228003436005),
            "10": (1.111420406956693, 1, 0.3519978321919493),
        },
        "D50": {
            "2": (0.9642119944211994, 1, 0.8251882845188288),
            "10": (0.9672062750333777, 1, 0.8142801513128616),
        },
        "D55": {
            "2": (0.956797052643698, 1, 0.9214805860173273),
            "10": (0.9579665682254781, 1, 0.9092525159847462),
        },
        "D65": {
            "2": (0.95047, 1.0, 1.08883),
            "10": (0.94809667673716, 1, 1.0730513595166162),
        },
        "D75": {
            "2": (0.9497220898840717, 1, 1.226393520724154),
            "10": (0.9441713925645873, 1, 1.2064272211720228),
        },
        "E": {"2": (1.0, 1.0, 1.0), "10": (1.0, 1.0, 1.0)},
    }
    coords = tf.constant(illuminants[illuminant.upper()][observer], input.dtype)

    xyz = xyz * coords

    return xyz_to_rgb(xyz)


def rgb_to_xyz(input, name=None):
    """
    Convert a RGB image to CIE XYZ.
    Args:
    input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    name: A name for the operation (optional).
    Returns:
    A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    input = tf.convert_to_tensor(input)
    assert input.dtype in (tf.float16, tf.float32, tf.float64)

    kernel = tf.constant(
        [
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227],
        ],
        input.dtype,
    )
    value = tf.where(
        tf.math.greater(input, 0.04045),
        tf.math.pow((input + 0.055) / 1.055, 2.4),
        input / 12.92,
    )
    return tf.tensordot(value, tf.transpose(kernel), axes=((-1,), (0,)))


def xyz_to_rgb(input, name=None):
    """
    Convert a CIE XYZ image to RGB.
    Args:
    input: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    name: A name for the operation (optional).
    Returns:
    A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    input = tf.convert_to_tensor(input)
    assert input.dtype in (tf.float16, tf.float32, tf.float64)

    # inv of:
    # [[0.412453, 0.35758 , 0.180423],
    #  [0.212671, 0.71516 , 0.072169],
    #  [0.019334, 0.119193, 0.950227]]
    kernel = tf.constant(
        [
            [3.24048134, -1.53715152, -0.49853633],
            [-0.96925495, 1.87599, 0.04155593],
            [0.05564664, -0.20404134, 1.05731107],
        ],
        input.dtype,
    )
    value = tf.tensordot(input, tf.transpose(kernel), axes=((-1,), (0,)))
    value = tf.where(
        tf.math.greater(value, 0.0031308),
        tf.math.pow(value, 1.0 / 2.4) * 1.055 - 0.055,
        value * 12.92,
    )
    return tf.clip_by_value(value, 0, 1)


# see: https://github.com/JetsonHacksNano/CSI-Camera/blob/master/simple_camera.py
def gstreamer_pipeline(
    capture_width,
    capture_height,
    display_width,
    display_height,
    framerate=10,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def get_RGB(L, AB):
    # Remove normalization
    L = (L + 1)*50
    AB = ((AB - 1)*255/2)+128

    LAB = tf.concat([L, AB], 3)
    #rgb = tfio.experimental.color.lab_to_rgb(LAB)
    rgb = lab_to_rgb(LAB)

    return rgb


def to_LAB(frame):
    # recast from int to float
    gray = tf.cast(frame, tf.float32)
    # normalize
    gray = gray/255

    gray = rgb_to_lab(gray)
    # "split" the array
    L, AB = gray[:,:,0], tf.stack([gray[:,:,1], gray[:,:,2]], axis=2)
    L = tf.reshape(L, shape=(256,256,1))
    # normalize
    L, AB = ((L/50.0) - 1., 1 + (2*(AB - 128)/255))
    return L, AB

def suitable_rgb(L, ab_img):
    # CIE LAB color space to RGB
    recolored_image = get_RGB(L, ab_img)
    # remove batch dimension
    recolored_image = tf.squeeze(recolored_image, 0)
    # remove normalization and interpret as uint8
    recolored_image = tf.cast(recolored_image*255, tf.uint8)
    # make it a numpy array
    recolored_image = recolored_image.numpy()
    # convert to BGR
    recolored_image = cv.cvtColor(recolored_image, cv.COLOR_RGB2BGR)
    return recolored_image
    
    
def create_Plot(loss_array):
    # set plot configuration
    plt.grid(True)
    plt.ylim(0, 0.07)
    plt.xlabel('Each Fram')
    plt.ylabel('Loss')
    
    plt.plot(loss_array)
    plt.savefig('loss.png')
    plt.clf() 
    

def saturate(image):
    hsvImg = cv.cvtColor(image,cv.COLOR_BGR2HSV)
    #multiple by a factor to change the saturation
    hsvImg[...,1] = hsvImg[...,1]*1.4
    #multiple by a factor of less than 1 to reduce the brightness 
    hsvImg[...,2] = hsvImg[...,2]*0.6
    image=cv.cvtColor(hsvImg,cv.COLOR_HSV2BGR)
    return image