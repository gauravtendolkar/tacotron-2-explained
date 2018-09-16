import tensorflow as tf


def composite_loss(mels, residual_added_mels, targets):
    mel_loss = tf.losses.mean_squared_error(targets, mels)
    residual_added_mel_loss = tf.losses.mean_squared_error(targets, residual_added_mels)
    loss = 0.5*mel_loss + 0.5*residual_added_mel_loss
    tf.summary.scalar("Training Loss", loss)
    return loss