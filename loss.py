import tensorflow as tf

@tf.function
def wce_loss(logits, labels, weights):
    """Computes the weighted cross-entropy loss


    Args:
        logits: tensor, output from final layer of model
        labels: tensor, labels
        weights: tensor, weights
    Returns:
        loss: scalar, weighted cross-entropy loss
    """

    labels = tf.cast(labels, dtype=tf.int32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    loss = tf.reduce_mean(loss*weights)

    return loss