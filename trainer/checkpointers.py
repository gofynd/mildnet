import csv
import six

import time
import warnings
import io

from collections import OrderedDict
from collections import Iterable

from .utils import *
from .accuracy import *
from time import time
from tensorflow.python.lib.io import file_io
from .datagen import ImageDataGeneratorCustom
import logging


class CSVLogger(tf.keras.callbacks.Callback):
    """Callback that streams epoch results to a csv file.
    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.
    # Example
    ```python
    csv_logger = CSVLogger('training.log')
    model.fit(X_train, Y_train, callbacks=[csv_logger])
    ```
    # Arguments
        filename: filename of the csv file, e.g. 'run/log.csv'.
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """

    def __init__(self, job_dir, filepath, separator=',', append=False):
        self.sep = separator
        self.append = append
        self.filepath = filepath
        self.writer = None
        self.keys = None
        self.job_dir = job_dir
        self.append_header = True
        if six.PY2:
            self.file_flags = 'b'
            self._open_args = {}
        else:
            self.file_flags = ''
            self._open_args = {'newline': '\n'}
        super(CSVLogger, self).__init__()

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filepath):
                with open(self.filepath, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            mode = 'a'
        else:
            mode = 'w'
        self.csv_file = io.open(self.filepath,
                                mode + self.file_flags,
                                **self._open_args)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k] if k in logs else 'NA') for k in self.keys])

        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep
            fieldnames = ['epoch'] + self.keys
            if six.PY2:
                fieldnames = [unicode(x) for x in fieldnames]
            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=fieldnames,
                                         dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

        if self.job_dir.startswith("gs://"):
            if os.path.exists(self.filepath):
                backup_file(self.job_dir, self.filepath)

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None
        
        
class ModelCheckpoint(tf.keras.callbacks.Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, job_dir, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.job_dir = job_dir
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            logging.info('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            logging.info('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    logging.info('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
        if self.job_dir.startswith("gs://"):
            if os.path.exists(filepath): 
                backup_file(self.job_dir, filepath)


class HyperdashCallback(tf.keras.callbacks.Callback):
    """_KerasCallback implement KerasCallback using an injected Experiment.
    
    # TODO: Decide if we want to handle the additional callbacks:
    # 1) on_epoch_begin
    # 2) on_batch_begin
    # 3) on_batch_end
    # 4) on_train_begin
    # 5) on_train_end
    """
    def __init__(self, exp):
        super(HyperdashCallback, self).__init__()
        self.last_time = time()
        self._exp = exp
    
    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            logs = {}
        acc = logs.get("accuracy")
        loss = logs.get("loss")
        val_acc = logs.get("val_accuracy")
        val_loss = logs.get("val_loss")

        if acc is not None:
            self._exp.metric("accuracy", acc)
        if loss is not None:
            self._exp.metric("loss", loss)
        if val_acc is not None:
            self._exp.metric("val_accuracy", val_acc)
        if val_loss is not None:
            self._exp.metric("val_loss", val_loss)
        self._exp.metric("mins_per_epoch", (time()-self.last_time)//60)
        self.last_time = time()
        self._exp.metric("current_epoch", epoch+1)


class TestAccuracy(tf.keras.callbacks.Callback):

    class DataGenerator(object):
        def __init__(self, params, data_path, target_size=(224, 224)):
            self.params = params
            self.target_size = target_size
            self.idg = ImageDataGeneratorCustom(**params)
            self.data_path = data_path

        def get_test_generator(self, batch_size):
            if not os.path.exists('tops_test.csv'):
                with file_io.FileIO(self.data_path + 'tops_test.csv', mode='r') as input_f:
                    with file_io.FileIO('tops_test.csv', mode='w+') as output_f:
                        output_f.write(input_f.read())
            return self.idg.flow_from_directory("tops/",
                                                batch_size=batch_size,
                                                target_size=self.target_size, shuffle=False,
                                                triplet_path='tops_test.csv')

    def __init__(self, data_path):
        self.data_path = data_path
        super(TestAccuracy, self).__init__()

    def on_train_begin(self, logs={}):
        dg = self.DataGenerator({
            "rescale": 1. / 255,
            "horizontal_flip": True,
            "vertical_flip": True,
            "zoom_range": 0.2,
            "shear_range": 0.2,
            "rotation_range": 30,
            "fill_mode": 'nearest' 
          }, self.data_path, target_size=(224, 224))
          
        self.test_generator = dg.get_test_generator(48)
        

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict_generator(self.test_generator, steps = self.test_generator.n//(self.test_generator.batch_size*10))
        acc = accuracy_fn(48)([], y_pred)
        logging.info("Test Accuracy: "+str(acc))
        return acc