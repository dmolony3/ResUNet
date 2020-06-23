import tensorflow as tf
import tensorflow_addons as tfa
import os
import random

class DataReader():
    """Class for reading images and generating batches of images"""
    def __init__(self, filepath, config):
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.use_weights = config.use_weights
        self.mirror = config.mirror
        self.rotate = config.rotate
        self.noise = config.noise
        self.image_size = config.input_size
        self.read_files(filepath)

    @property
    def num_images(self):
        """Returns the number of files in the training set"""

        return len(self.image_list)

    def read_files(self, data_file):
        """Reads file contanining paths to image, labels and weights

		The input data_file is a text file where each line of the file 
		contains the path to the image and the label separated by a comma.
		Additionally it may also contain the path to a weighted image. The 
		images, labels and weights are added to lists


		Args:
			data_file: path to the data_file
		"""

        f = open(data_file, 'r')
        data = f.read()
        f.close()
        data = data.split('\n')
        image_list = []
        label_list = []
        weight_list = []

        for i in range(len(data)):
            line = data[i]
            if line:
                if self.use_weights:
                    try:
                        image, label, weight = line.split(',')
                    except ValueError:
                        print("Use weights flag is turned on, check that the \
                        file contains comma separated lines for image, label and weight")                    
                    image_list.append(image)
                    label_list.append(label)
                    weight_list.append(weight)
                else:
                    try:
                        image, label = line.split(',')
                    except ValueError:
                        image = line
                        label = None
                    image_list.append(image)
                    label_list.append(label)

        self.image_list = image_list
        self.label_list = label_list
        self.weight_list = weight_list
    
    def decode_image(self, image, label, weight):
        """function that reads an image and decode to a tensor"""

        image = tf.io.read_file(image)
        image = tf.io.decode_jpeg(image, channels=3)

        if self.use_weights:
            label = tf.io.read_file(label)
            weight = tf.io.read_file(weight)
            label = tf.io.decode_png(label)
            weight = tf.io.decode_png(weight)
        else:
            if label is not None:
                label = tf.io.read_file(label)
                label = tf.io.decode_png(label)
            else:
                label = tf.zeros_like(image[:, :, 0])
            weight = tf.zeros_like(image[:, :, 0])

        label = tf.squeeze(label, axis=-1)
        weight = tf.squeeze(weight, axis=-1)

        # convert image to float
        image = tf.cast(image, dtype=tf.float32)
        label = tf.cast(label, dtype=tf.int32)
        weight = tf.cast(weight, dtype=tf.float32)
        
        return image, label, weight

    def mirror_image(self, image, label, weight):
        """left to right flip with random probability"""

        label = tf.expand_dims(label, axis=-1)
        weight = tf.expand_dims(weight, axis=-1)

        cond = tf.cast(tf.random.uniform([], maxval=2, dtype=tf.int32), tf.bool)
        image = tf.cond(cond, lambda: tf.image.flip_left_right(image), 
                        lambda: tf.identity(image))
        label = tf.cond(cond, lambda: tf.image.flip_left_right(label), 
                        lambda: tf.identity(label))
        weight = tf.cond(cond, lambda: tf.image.flip_left_right(weight), 
                            lambda: tf.identity(weight))
        
        label = tf.squeeze(label, axis=-1)
        weight = tf.squeeze(weight, axis=-1)

        return image, label, weight

    def rotate_image(self, image, label, weight):
        """Rotate images"""

        rot_angle = tf.random.uniform([], minval=0, maxval=360, dtype=tf.float32)
        image = tfa.image.rotate(image, rot_angle)
        label = tfa.image.rotate(label, rot_angle)
        weight = tfa.image.rotate(weight, rot_angle)
        
        return image, label, weight
   
    def add_noise(self, image, label, weight):
        """Add gaussian noise"""

        noise = tf.random.normal(shape=tf.shape(image[:, :, 0]), mean=0.0, stddev=1)
        noise = noise=tf.stack([noise]*3, axis=2)
        image += noise
        
        return image, label, weight

    def resize(self, image, label, weight):
        """Resizes image to image size"""

        image = tf.image.resize(image, [self.image_size, self.image_size])
        label = tf.image.resize(tf.expand_dims(label, axis=-1), 
                                [self.image_size, self.image_size])
        weight = tf.image.resize(tf.expand_dims(weight, axis=-1), 
                                [self.image_size, self.image_size])

        label = tf.squeeze(label, axis=-1)
        weight = tf.squeeze(weight, axis=-1)

        return image, label, weight

    def read_batch(self, train, num_epochs, shuffle=False):
        """Returns batch of images
        
        Args:
            train: flag indicating whether in training mode for data augmentation
        Returns:
            data: tuple, batch of image, label, weight, size and phenotype
        """

        data = tf.data.Dataset.from_tensor_slices((self.image_list, self.label_list, self.weight_list))

        if shuffle:
            data = data.shuffle(len(self.image_list))

        data = data.map(self.decode_image)
        data = data.map(self.resize)

        # Data augmentation
        if train:
            if self.rotate:
                data = data.map(self.rotate_image,  num_parallel_calls=2)
            if self.mirror:
                data = data.map(self.mirror_image, num_parallel_calls=2)
            if self.noise:
                data = data.map(self.add_noise,  num_parallel_calls=2)

        data = data.batch(batch_size=self.batch_size, drop_remainder=True)
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
        data = data.repeat(num_epochs)

        return data
