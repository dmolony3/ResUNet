import os
import tensorflow as tf
from data_reader import DataReader
from utils import accuracy
from PIL import Image as im

def predict(model, config):
    """Generates model prediction and writes to file

    Args:
        model: tensorflow keras model
        config: instance of class configuration
    """

    data = DataReader(config.val_file_path, config)
    batch = data.read_batch(train=False, num_epochs=1)

    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), net=model)
    manager = tf.train.CheckpointManager(checkpoint, config.save_directory, max_to_keep=10)

    # restore weights if they exist
    if manager.latest_checkpoint:
        try:
            checkpoint.restore(manager.latest_checkpoint)
            print('Restoring weights from {}'.format(manager.latest_checkpoint))
        except:
            print('No checkpoint found')
    else:
        print('No checkpoint found')
        

    for iteration, (images, _, _) in enumerate(batch):

        logits = model(images, training=False)
        logits = tf.image.resize(logits, [config.input_size, config.input_size])

        preds = tf.cast(tf.argmax(logits, axis=-1), dtype=tf.uint8)

        for j in range(preds.shape[0]):
            fname = str(iteration*config.batch_size + j) + '.png'
            img = im.fromarray(preds[j].numpy(), 'L')
            img.save(os.path.join(config.save_directory, 'predictions', fname))