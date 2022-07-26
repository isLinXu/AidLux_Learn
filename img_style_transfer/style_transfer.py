"""Import dependencies."""

import tensorflow as tf
import matplotlib.pyplot as plt
import pylab
import cv2

"""Pre-process the inputs"""


# Function to load an image from a file, and add a batch dimension.
def load_img(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]

    return img


# Function to pre-process by resizing an central cropping it.
def preprocess_image(image, target_dim):
    # Resize the image so that the shorter dimension becomes 256px.
    shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
    short_dim = min(shape)
    scale = target_dim / short_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    image = tf.image.resize(image, new_shape)

    # Central crop the image.
    image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)

    return image


# Function to run style prediction on preprocessed style image.
def run_style_predict(preprocessed_style_image, style_predict_path):
    # Load the model.
    interpreter = tf.lite.Interpreter(model_path=style_predict_path)

    # Set model input.
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]["index"], preprocessed_style_image)

    # Calculate style bottleneck.
    interpreter.invoke()
    style_bottleneck = interpreter.tensor(
        interpreter.get_output_details()[0]["index"]
    )()

    return style_bottleneck


# Run style transform on preprocessed style image
def run_style_transform(style_bottleneck, preprocessed_content_image, style_transform_path):
    # Load the model.
    interpreter = tf.lite.Interpreter(model_path=style_transform_path)

    # Set model input.
    input_details = interpreter.get_input_details()
    interpreter.allocate_tensors()

    # Set model inputs.
    interpreter.set_tensor(input_details[0]["index"], preprocessed_content_image)
    interpreter.set_tensor(input_details[1]["index"], style_bottleneck)
    interpreter.invoke()

    # Transform content image.
    stylized_image = interpreter.tensor(
        interpreter.get_output_details()[0]["index"]
    )()

    return stylized_image


def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


def main():
    """Load the content and style images, and the pre-trained TensorFlow Lite models."""
    content_path = './content_img/zcool.jpg'
    style_path = './style_img/airmb.jpg'
    style_predict_path = 'style_prediction_256_fp16.tflite'
    style_transform_path = 'style_transfer_256_fp16.tflite'

    # Load the input images.
    content_image = load_img(content_path)
    style_image = load_img(style_path)

    # Preprocess the input images.
    preprocessed_content_image = preprocess_image(content_image, 384)
    preprocessed_style_image = preprocess_image(style_image, 256)

    print('Style Image Shape:', preprocessed_style_image.shape)
    print('Content Image Shape:', preprocessed_content_image.shape)

    # Calculate style bottleneck for the preprocessed style image.
    style_bottleneck = run_style_predict(preprocessed_style_image, style_predict_path)
    print('Style Bottleneck Shape:', style_bottleneck.shape)

    # Stylize the content image using the style bottleneck.
    stylized_image = run_style_transform(style_bottleneck, preprocessed_content_image, style_transform_path)

    # Visualize the output.

    imshow(stylized_image, 'Stylized Image')
    plt.savefig('Stylized Image.jpg')
    plt.show()

    print(stylized_image)

if __name__ == '__main__':
    main()
