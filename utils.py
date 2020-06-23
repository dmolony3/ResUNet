import tensorflow as tf
import math

@tf.function
def accuracy(labels, preds, num_classes):
    """Determine intersection over union
    
    Args:
        labels: tensor, labelmap, each pixel is an image class.
        preds: tensor, predicted labels, each pixel is an class.
        num_classes: int, number of classes to predict
    Returns:
        IOU: Intersection over union i.e. Jaccard index
    """

    IOU = []
    for i in range(num_classes):
        inter = tf.math.multiply(tf.cast(tf.math.equal(labels, i), dtype=tf.int32), 
                                 tf.cast(tf.math.equal(preds, i), dtype=tf.int32))
        inter = tf.reduce_sum(inter, axis=(1,2))
        union = tf.subtract(tf.add(tf.reduce_sum(tf.cast(tf.math.equal(labels, i), dtype=tf.int32), axis=(1,2)), 
                                  tf.reduce_sum(tf.cast(tf.math.equal(preds, i), dtype=tf.int32), axis=(1,2))), inter)
        IOU.append(inter/union)
    IOU = tf.stack(IOU, axis=1)
    IOU = tf.reduce_mean(IOU, 0)

    return IOU

class LinearWarmUpCosineDecay():
    def __init__(self, total_iterations, learning_rate):
        """Updates the learning rate with a linear warmup and a cosine decay
        
        Args:
            total_iterations: the total iterations the model will run for
            learning_rate: initial learning rate
        Attributes:
            warmup_iterations: number of iterations for linear warmup
            learning_rate_min: minimum allowed value for learning rate
            total_iterations: the total iterations the model will run for
            learning_rate: initial learning rate
        Returns:
            learning_rate: learning rate for current iteration
        """

        self.warmup_iterations = 0
        self.learning_rate_min = 0
        self.learning_rate = learning_rate
        self.total_iterations = total_iterations

    def __call__(self, current_iteration):
        if self.warmup_iterations > 0 and current_iteration <= self.warmup_iterations:
            learning_rate = self.learning_rate*(current_iteration/self.warmup_iterations)
        else:
            learning_rate = self.learning_rate_min + 0.5*(self.learning_rate - self.learning_rate_min)*(1+tf.cos(current_iteration/(self.total_iterations)*math.pi))

        return learning_rate   
