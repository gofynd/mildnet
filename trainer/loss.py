from tensorflow.keras import backend as K
import tensorflow as tf

_EPSILON = K.epsilon()


def hinge_loss_fn(batch_size):
    def hinge_loss(y_true, y_pred):
        y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
        loss = tf.convert_to_tensor(0,dtype=tf.float32)
        g = tf.constant(1.0, shape=[1], dtype=tf.float32)
        for i in range(0, batch_size, 3):
            try:
                q_embedding = y_pred[i+0]
                p_embedding = y_pred[i+1]
                n_embedding = y_pred[i+2]
                D_q_p =  K.sqrt(K.sum((q_embedding - p_embedding)**2))
                D_q_n = K.sqrt(K.sum((q_embedding - n_embedding)**2))
                loss = (loss + g + D_q_p - D_q_n)            
            except:
                continue
        loss = loss/(batch_size/3)
        zero = tf.constant(0.0, shape=[1], dtype=tf.float32)
        return tf.maximum(loss,zero)
    return hinge_loss


def hinge_new_loss_fn(batch_size):
    def hinge_new_loss(y_true, y_pred):
        y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
        loss = tf.convert_to_tensor(0,dtype=tf.float32)
        g = tf.constant(1.0, shape=[1], dtype=tf.float32)
        for i in range(0, batch_size, 3):
            try:
                q_embedding = y_pred[i+0]
                p_embedding = y_pred[i+1]
                n_embedding = y_pred[i+2]
                D_q_p =  K.sqrt(K.sum((q_embedding - p_embedding)**2))
                D_q_n = K.sqrt(K.sum((q_embedding - n_embedding)**2))
                D_p_n = K.sqrt(K.sum((p_embedding - n_embedding)**2))
                loss = (loss + g + D_q_p - D_q_n + D_q_p - D_p_n)            
            except:
                continue
        loss = loss/(batch_size/6)
        zero = tf.constant(0.0, shape=[1], dtype=tf.float32)
        return tf.maximum(loss,zero)
    return hinge_new_loss


def hinge_twice_loss_fn(batch_size):
    def hinge_twice_loss(y_true, y_pred):
        y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
        loss = tf.convert_to_tensor(0,dtype=tf.float32)
        g = tf.constant(1.0, shape=[1], dtype=tf.float32)
        for i in range(0, batch_size, 3):
            try:
                q_embedding = y_pred[i+0]
                p_embedding = y_pred[i+1]
                n_embedding = y_pred[i+2]
                D_q_p =  K.sqrt(K.sum((q_embedding - p_embedding)**2))
                D_q_n = K.sqrt(K.sum((q_embedding - n_embedding)**2))
                loss = (loss + g + D_q_p - D_q_n)          
            except:
                continue
        loss = loss/(batch_size/6)
        zero = tf.constant(0.0, shape=[1], dtype=tf.float32)
        return tf.maximum(loss,zero)
    return hinge_twice_loss


def contrastive_loss_fn(batch_size):
    def contrastive_loss(y_true, y_pred):
        def _contrastive_loss(y1, D):
            g = tf.constant(1.0, shape=[1], dtype=tf.float32)
            return K.mean(y1 * K.square(D) +
                          (g - y1) * K.square(K.maximum(g - D, 0)))

        y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
        loss = tf.convert_to_tensor(0,dtype=tf.float32)
        g = tf.constant(1.0, shape=[1], dtype=tf.float32)
        h = tf.constant(0.0, shape=[1], dtype=tf.float32)
        for i in range(0,batch_size,3):
            try:
                q_embedding = y_pred[i+0]
                p_embedding = y_pred[i+1]
                n_embedding = y_pred[i+2]
                D_q_p = K.sqrt(K.sum((q_embedding - p_embedding)**2))
                D_q_n = K.sqrt(K.sum((q_embedding - n_embedding)**2))
                L_q_p = _contrastive_loss(g, D_q_p)
                L_q_n = _contrastive_loss(h, D_q_n)
                loss = (loss + L_q_p + L_q_n )
            except:
                continue
        loss = loss/(batch_size*2/3)
        zero = tf.constant(0.0, shape=[1], dtype=tf.float32)
        return tf.maximum(loss,zero)
    return contrastive_loss


#https://towardsdatascience.com/lossless-triplet-loss-7e932f990b24
def lossless_loss_fn(batch_size):
    def lossless_loss(y_true, y_pred):
        N = tf.constant(4096.0, shape=[1], dtype=tf.float32)
        beta = tf.constant(4096.0, shape=[1], dtype=tf.float32)
        y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
        loss =  tf.convert_to_tensor(0,dtype=tf.float32)
        g = tf.constant(1.0, shape=[1], dtype=tf.float32)
        const1 = tf.constant(1.0, shape=[1], dtype=tf.float32)
        for i in range(0,batch_size,3):
            try:
                anchor = y_pred[i+0]
                positive =  y_pred[i+1]
                negative = y_pred[i+2]

                pos_dist = K.sum(K.square(anchor-positive),1)
                neg_dist = K.sum(K.square(anchor,negative),1)

                pos_dist = -tf.log(-tf.divide((pos_dist), beta)+const1+epsilon)
                neg_dist = -tf.log(-tf.divide((N-neg_dist), beta)+const1+epsilon)

                _loss = neg_dist + pos_dist

                loss = (loss + g + _loss)
            except:
                continue
        loss = loss/(batch_size/3)
        zero = tf.constant(0.0, shape=[1], dtype=tf.float32)
        return tf.maximum(loss,zero)
    return lossless_loss

  
def angular_loss_1_fn(batch_size):
    def angular_loss_1(y_true, y_pred):
        y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
        loss =  tf.convert_to_tensor(0,dtype=tf.float32)
        g = tf.constant(1.0, shape=[1], dtype=tf.float32)
        c = tf.constant(4.0, shape=[1], dtype=tf.float32)
        alpha = tf.constant(45.0, shape=[1], dtype=tf.float32)

        for i in range(0,batch_size,3):
            try:
                xa = y_pred[i+0]
                xp =  y_pred[i+1]
                xn = y_pred[i+2]

                sq =  K.square(xa-xp)
                xc = (xa+xp)/2
                _loss = sq - c*(tf.tan(alpha*K.square(xn-xc))**2)
                zero = tf.constant(0.0, shape=[1], dtype=tf.float32)
                _loss = tf.maximum(_loss,zero)

                loss = (loss + g + _loss)
            except:
                continue
        loss = loss/(batch_size/3)
        return loss
    return angular_loss_1


def angular_loss_2_fn(batch_size):
    def angular_loss_2(y_true, y_pred):
        y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
        loss =  tf.convert_to_tensor(0,dtype=tf.float32)
        g = tf.constant(1.0, shape=[1], dtype=tf.float32)
        c = tf.constant(4.0, shape=[1], dtype=tf.float32)
        d = tf.constant(2.0, shape=[1], dtype=tf.float32)
        alpha = tf.constant(45.0, shape=[1], dtype=tf.float32)

        losses = []
        losses2 = []
        for i in range(0,batch_size,3):
            try:
                xa = y_pred[i+0]
                xp =  y_pred[i+1]
                xn = y_pred[i+2]

                fapn = c*(tf.tan(alpha*K.transpose(xa+xp)*xn)**2) - d*(g+tf.tan(alpha)**2)*K.transpose(xa)*xp
                losses.append(fapn)
                
                losses2.append(K.transpose(xa)*xn - K.transpose(xa)*xp)

                loss = (loss + g + _loss)
            except:
                continue
        loss = K.sum(K.log(1+2*K.sum([K.exp(v) for v in losses])))
        loss2 = K.sum(K.log(1+2*K.sum([K.exp(v) for v in losses2])))
        loss = loss + 2*loss2
        loss = loss/(batch_size/3)
        zero = tf.constant(0.0, shape=[1], dtype=tf.float32)
        return tf.maximum(loss,zero)
    return angular_loss_2