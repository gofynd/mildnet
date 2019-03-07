from tensorflow.keras import backend as K
import tensorflow as tf

_EPSILON = K.epsilon()

def accuracy_fn(batch_size):

    def accuracy(y_true, y_pred):
        y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
        accuracy = 0
        for i in range(0,batch_size,3):
            try:
                q_embedding = y_pred[i+0]
                p_embedding = y_pred[i+1]
                n_embedding = y_pred[i+2]
                D_q_p =  K.sqrt(K.sum((q_embedding - p_embedding)**2))
                D_q_n = K.sqrt(K.sum((q_embedding - n_embedding)**2))
                accuracy+=tf.cond(D_q_n > D_q_p, lambda: 1, lambda: 0)
            except:
                continue
        accuracy = tf.cast(accuracy, tf.float32)
        return accuracy*100/(batch_size/3)

    return accuracy