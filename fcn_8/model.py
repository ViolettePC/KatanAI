from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Dropout, Conv2DTranspose, Add


def fcn8_model(img_size, input_channels=1, n_classes=1, initial_features=16, kernel_size=(3, 3)):
    """
    :param img_size: Tuple containing the shape of input images in pixels (width, height).
    :param input_channels: Number (Int) of channels in the input images (1 for grayscale images).
    :param n_classes: Number of class to segment.
    :param initial_features: Number (Int) of output features after the first convolution layer.
    :param kernel_size: Size (Int) of the kernel used in encoder's convolutions.
    :return: Model.
    """

    """ Instantiation of the input tensor (input_1):"""
    inputs = Input(shape=(img_size[0], img_size[1], input_channels))
    x = inputs

    """ Downsampling """
    """
    First downsampling block containing 2 convolution layers followed by a Max pooling layer. 
    """
    x = Conv2D(filters=initial_features, kernel_size=kernel_size, activation='elu', padding='same')(x)
    x = Conv2D(initial_features, kernel_size=kernel_size, activation='elu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    """
    Second downsampling block containing 2 convolution layers followed by a Max pooling layer. 
    """
    x = Conv2D(initial_features * 2, kernel_size=kernel_size, activation='elu', padding='same')(x)
    x = Conv2D(initial_features * 2, kernel_size=kernel_size, activation='elu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    """
    Third downsampling block containing 3 convolution layers followed by a Max pooling layer. 
    """
    x = Conv2D(initial_features * 4, kernel_size=kernel_size, activation='elu', padding='same')(x)
    x = Conv2D(initial_features * 4, kernel_size=kernel_size, activation='elu', padding='same')(x)
    x = Conv2D(initial_features * 4, kernel_size=kernel_size, activation='elu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    """ Save the Max pooling's output to feed a skip connection"""
    pool3 = x

    """
    Fourth downsampling block containing 3 convolution layers followed by a Max pooling layer. 
    """
    x = Conv2D(initial_features * 8, kernel_size=kernel_size, activation='elu', padding='same')(x)
    x = Conv2D(initial_features * 8, kernel_size=kernel_size, activation='elu', padding='same')(x)
    x = Conv2D(initial_features * 8, kernel_size=kernel_size, activation='elu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    """ Save the Max pooling's output to feed a skip connection"""
    pool4 = x

    """
    Fifth downsampling block containing 3 convolution layers followed by a Max pooling layer. 
    """
    x = Conv2D(initial_features * 8, kernel_size=kernel_size, activation='elu', padding='same')(x)
    x = Conv2D(initial_features * 8, kernel_size=kernel_size, activation='elu', padding='same')(x)
    x = Conv2D(initial_features * 8, kernel_size=kernel_size, activation='elu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    """ Save the Max pooling's output to feed a skip connection"""
    pool5 = x

    """
    Continue feature extraction though 2 additional convolution + dropout layers.
    """
    conv6 = Conv2D(2048, (7, 7), activation='elu', padding='same')(x)
    conv6 = Dropout(0.5)(conv6)
    conv7 = Conv2D(2048, (1, 1), activation='elu', padding='same')(conv6)
    conv7 = Dropout(0.5)(conv7)

    """ Convolve the output of the fourth Max pooling. """
    pool4_n = Conv2D(n_classes, (1, 1), activation='elu', padding='same', name='pool4_n')(pool4)
    """ First transposed convolution. """
    u2 = Conv2DTranspose(n_classes, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv7)
    """ Addition of the previous transposed convolution's output with the convolved fourth Max pooling. """
    u2_skip = Add()([pool4_n, u2])

    """ Convolve the output of the third Max pooling. """
    pool3_n = Conv2D(n_classes, (1, 1), activation='elu', padding='same', name='pool3_n')(pool3)
    """ Second transposed convolution. """
    u4 = Conv2DTranspose(n_classes, kernel_size=(2, 2), strides=(2, 2), padding='same')(u2_skip)
    """ Addition of the previously transposed convolution's output with the convolved third Max pooling."""
    u4_skip = Add()([pool3_n, u4])

    """ Third and last transposed convolution followed by a sigmoid activation. """
    x = Conv2DTranspose(n_classes, kernel_size=(8, 8), strides=(8, 8), padding='same',
                        activation='sigmoid')(u4_skip)

    model = Model(inputs=[inputs], outputs=[x], name=f'FCN-8-F{initial_features}')

    return model
