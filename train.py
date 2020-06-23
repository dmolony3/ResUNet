import os
import tensorflow as tf
from data_reader import DataReader
from loss import wce_loss
from utils import accuracy, LinearWarmUpCosineDecay

@tf.function
def train_step(model, images, labels, weights):
    """Performes one training step

    Args:
        model: tensorflow keras model
        images: tensor, batch of input images
        labels: tensor, batch of image labels
        weights: tensor, batch of image weights
    Returns:
        loss: model loss
        grads: gradient with respect to model variables
        preds: tensor, class label prediction
    """

    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        logits = tf.image.resize(logits, tf.shape(labels)[1:3])
        loss = wce_loss(logits, labels, weights)
        reg_loss = tf.add_n(model.losses) if model.losses else 0
        loss = loss + reg_loss
    
    grads = tape.gradient(loss, model.trainable_variables)
    preds = tf.argmax(logits, axis=-1)

    return loss, grads, preds

@tf.function
def val_step(model, images, labels):
    """Performs one validation step

    Args:
        model: tensorflow keras model
        images: tensor, batch of input images
        labels: tensor, batch of image labels
    Returns:
        loss: model loss
        preds: tensor, class label prediction
    """

    logits = model(images, training=False)
    logits = tf.image.resize(logits, tf.shape(labels)[1:3])
    loss = wce_loss(logits, labels, tf.ones_like(labels))
    reg_loss = tf.add_n(model.losses) if model.losses else 0
    loss = loss + reg_loss   

    preds = tf.argmax(logits, axis=-1)

    return loss, preds

def train(model, config):
    """Trains the input model using specified configurations

    Args:
        model: tensorflow keras model
        config: instance of class configuration
    """

    train_data = DataReader(config.train_file_path, config)
    train_batch = train_data.read_batch(train=True, num_epochs=config.num_epochs, 
                                        shuffle=True)
    train_iterations = int(train_data.num_images//config.batch_size)

    if config.val_file_path:
        val_data = DataReader(config.val_file_path, config)

    learning_rate = LinearWarmUpCosineDecay(train_iterations*config.num_epochs,
                                            config.learning_rate)   
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate(0))


    epoch = 1
    epoch_loss_train = 0
    for iteration, (images, labels, weights) in enumerate(train_batch):
        loss, grads, preds = train_step(model, images, labels, weights)
        epoch_loss_train += loss

        optimizer.__setattr__('lr', learning_rate(optimizer.iterations))
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if iteration >  0 and iteration % train_iterations == 0:
            print("Epoch {} Train loss:  {}".format(epoch, epoch_loss_train/train_iterations))
            epoch_loss_train = 0

            if config.val_file_path:
                epoch_loss_val = []
                acc = []

                val_batch = val_data.read_batch(train=False, num_epochs=1)

                for images, labels, weights in val_batch:
                    loss, preds = val_step(model, images, labels)
                    epoch_loss_val.append(loss)

                    acc.append(accuracy(labels, preds, config.num_classes))

                print("Epoch {} Val loss:  {}".format(epoch, epoch_loss_val/len(epoch_loss_val)))

                for j in config.num_classes:
                    print("Epoch {} Class {} Accuracy: {}".format(epoch, j, sum([val[j] for val in acc])/len(acc)))

            model.save_weights(os.path.join(config.save_directory, 'model'), 
                               save_format='tf')
            epoch += 1