import os
import traceback
import tensorflow as tf
import numpy as np
import json
from nets.resnet_at_cifar10 import ModelHelper
from learners.discr_channel_pruning.learner import DisChnPrunedLearner

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', './logs', 'logging directory')
tf.app.flags.DEFINE_boolean('enbl_multi_gpu', False, 'enable multi-GPU training')
tf.app.flags.DEFINE_string('exec_mode', 'train', 'execution mode: train / eval')
tf.app.flags.DEFINE_boolean('debug', False, 'debugging information')

def main(unused_argv):
  # Monkeypatching for Valohai JSON log output
  DisChnPrunedLearner._DisChnPrunedLearner__monitor_progress = __monitor_progress
  DisChnPrunedLearner.evaluate = evaluate

  try:
    tf.logging.set_verbosity(tf.logging.INFO)
    sm_writer = tf.summary.FileWriter(FLAGS.log_dir)
    model_helper = ModelHelper()
    learner = DisChnPrunedLearner(sm_writer, model_helper)
    learner.train()
    return 0
  except ValueError:
    traceback.print_exc()
    return 1

def evaluate(self):
  save_path = tf.train.latest_checkpoint(os.path.dirname(FLAGS.dcp_save_path))
  self.saver_prnd_eval.restore(self.sess_eval, save_path)
  tf.logging.info('model restored from ' + save_path)
  nb_iters = int(np.ceil(float(FLAGS.nb_smpls_eval) / FLAGS.batch_size))
  eval_rslts = np.zeros((nb_iters, len(self.eval_op)))
  for idx_iter in range(nb_iters):
    eval_rslts[idx_iter] = self.sess_eval.run(self.eval_op)
    print(json.dumps({"accuracy": np.asscalar(eval_rslts[idx_iter, 3]), "mean": np.asscalar(np.mean(eval_rslts[0:idx_iter, 3]))}))

def __monitor_progress(self, summary, log_rslt, idx_iter, time_step):
  speed = FLAGS.batch_size * FLAGS.summ_step / time_step
  print(json.dumps({"iteration": idx_iter + 1, "speed": speed, "training_accuracy": np.asscalar(log_rslt[4]), "loss": np.asscalar(log_rslt[1])}))

if __name__ == '__main__':
  tf.app.run()
