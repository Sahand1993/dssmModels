from DSSM.dssm.model import *

saver = tf.compat.v1.train.Saver()

with tf.compat.v1.Session() as sess:
  init = tf.compat.v1.global_variables_initializer()
  sess.run(init)
  path = saver.save(sess, "./saved_model.ckpt", global_step=1)
  print(path)

with tf.compat.v1.Session() as sess:
  a = tf.compat.v1.train.latest_checkpoint(".")
  print(a)
  saver.restore(sess, "./saved_model.ckpt-0")