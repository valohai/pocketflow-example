import os
import traceback
import tensorflow as tf
import numpy as np
import json

from nets.resnet_at_cifar10 import ModelHelper
from learners.full_precision.learner import FullPrecLearner

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', './logs', 'logging directory')
tf.app.flags.DEFINE_boolean('enbl_multi_gpu', False, 'enable multi-GPU training')
tf.app.flags.DEFINE_string('exec_mode', 'eval', 'execution mode: train / eval')
tf.app.flags.DEFINE_boolean('debug', False, 'debugging information')

def main(unused_argv):
  # Monkeypatching for Valohai JSON log output
  FullPrecLearner.evaluate = evaluate
  
  try:
    tf.logging.set_verbosity(tf.logging.INFO)
    sm_writer = tf.summary.FileWriter(FLAGS.log_dir)
    model_helper = ModelHelper()
    learner = FullPrecLearner(sm_writer, model_helper)
    learner.evaluate()

    return 0
  except ValueError:
    traceback.print_exc()
    return 1

def evaluate(self):
  save_path = tf.train.latest_checkpoint('./models')
  self.saver_eval.restore(self.sess_eval, save_path)
  tf.logging.info('model restored from ' + save_path)
  nb_iters = int(np.ceil(float(FLAGS.nb_smpls_eval) / FLAGS.batch_size))
  eval_rslts = np.zeros((nb_iters, len(self.eval_op)))
  for idx_iter in range(nb_iters):
    eval_rslts[idx_iter] = self.sess_eval.run(self.eval_op)
    print(json.dumps({"accuracy": np.asscalar(eval_rslts[idx_iter, 1]), "mean": np.asscalar(np.mean(eval_rslts[0:idx_iter, 1]))}))

if __name__ == '__main__':
  tf.app.run()
