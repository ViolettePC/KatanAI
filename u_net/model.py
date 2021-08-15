from tensorflow import keras


def unet_model(img_size, n_levels, input_channels=1, output_channels=1,
               initial_features=32, nb_sub_blocks=2, kernel_size=3, pooling_size=2, batch_norm=True):
    """
    :param img_size: Tuple containing the shape of input images in pixels (width, height).
    :param n_levels: Number (Int) of blocks in the encoder (same number for the decoder).
    :param input_channels: Number (Int) of channels in the input images (1 for grayscale images).
    :param output_channels: Number (Int) of channels in the input images (match input_channels).
    :param initial_features: Number (Int) of output features after the first convolution layer.
    :param nb_sub_blocks: Number (Int) of (2D conv + BN + ReLU) in a block.
    :param kernel_size: Size (Int) of the kernel used in every convolution layers (except the last one).
    :param pooling_size: Size (Int) of kernel used in pooling operations (divide size of feature maps by n=pooling_size).
    :param batch_norm: Boolean to activate or deactivate every batch normalization layers.
    :return: Model.
    """

    """ Instantiation of the input tensor (input_1):"""
    inputs = keras.layers.Input(shape=(img_size[0], img_size[1], input_channels))
    x = inputs

    skip_connections = {}

    """ ENCODER + Bottleneck"""
    """ 
    Successive applications of 2D convolutions + Batch Normalization (optional) + ReLU.
    Each of those blocks are then followed by a 2D max pooling operation (Exception for Bottleneck).
    """
    for level in range(n_levels):
        for _ in range(nb_sub_blocks):
            x = keras.layers.Conv2D(filters=initial_features * 2 ** level, kernel_size=kernel_size, padding='same')(x)
            x = keras.layers.BatchNormalization()(x) if batch_norm else x
            x = keras.layers.Activation('relu')(x)
        if level < n_levels - 1:
            """ Skip connections: store the output feature maps of every blocks from the encoder. """
            skip_connections[level] = x
            """ 2D max pooling operation to downsample feature maps"""
            x = keras.layers.MaxPool2D(pooling_size)(x)

    """ DECODER """
    """ 
    Successive applications of 2D convolutions + Batch Normalization (optional) + ReLU.
    Each of those blocks are preceded by a 2D up-convolution operation.
    """
    for level in reversed(range(n_levels-1)):
        """ Up-convolution operation: """
        x = keras.layers.Conv2DTranspose(filters=initial_features * 2 ** level, strides=pooling_size,
                                         kernel_size=kernel_size, padding='same')(x)
        """ Concatenate the feature maps of the encoder with the corresponding feature maps of the decoder. """
        x = keras.layers.Concatenate()([x, skip_connections[level]])
        for _ in range(nb_sub_blocks):
            x = keras.layers.Conv2D(filters=initial_features * 2 ** level, kernel_size=kernel_size, padding='same')(x)
            x = keras.layers.BatchNormalization()(x) if batch_norm else x
            x = keras.layers.Activation('relu')(x)

    """ Final 2D 1x1 kernel size convolution layer to classify every pixels """
    x = keras.layers.Conv2D(output_channels, kernel_size=1, activation='sigmoid', padding='same')(x)

    return keras.Model(inputs=[inputs], outputs=[x], name=f'UNET-L{n_levels}-F{initial_features}')
