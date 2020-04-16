# 验证tensorflow
import tensorflow as tf
print(tf.__version__)
# 验证cuda
a = tf.test.is_built_with_cuda()
print(a)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
