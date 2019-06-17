import tensorflow as tf
import glob
import numpy as np
from datetime import datetime
from optparse import OptionParser
import pycuda.driver as cuda
import pycuda.autoinit
import sys, os
from preprocess import *
import ifx

# ================== calculate time
def millis(start_time):
  dt = datetime.now() - start_time
  ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds/1000.0
  return ms

# ================== tfrecord function helper
def eval_input_fn(data_files, preprocess_fn):
  dataset = tf.data.TFRecordDataset(data_files, num_parallel_reads=32, buffer_size=10)
  dataset = dataset.map(preprocess_fn)
  dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
  dataset = dataset.repeat(count=1)
  dataset = dataset.batch(batch_size=1)
  iterator = dataset.make_one_shot_iterator()
  features, labels = iterator.get_next()
  return features, labels, iterator

def deserialize_image_record(record):
  feature_map = {
      'image/encoded': tf.FixedLenFeature([], tf.string, ''),
      'image/class/label': tf.FixedLenFeature([1], tf.int64, -1),
  }
  
  obj = tf.parse_single_example(record, feature_map)
  imgdata = obj['image/encoded']
  label = tf.cast(obj['image/class/label'], tf.int32)
  return imgdata, label

def get_preprocess_fn():
  def process(record):
    imgdata, label = deserialize_image_record(record)
    image = preprocess_image(imgdata, INPUT_SIZE, INPUT_SIZE)
    return image, label

  return process

# ================== get param
parser = OptionParser() 
parser.add_option("-m", "--model", action="store", 
                  dest="model", 
                  help="input model file")
parser.add_option("-d", "--data_dir", action="store", 
                  dest="data_dir", 
                  help="imagenet evaluation tfrecord directory") 
(options, args) = parser.parse_args()

model_path = os.path.abspath(options.model)
imagenet_eval_dir = os.path.abspath(options.data_dir)

# ================== init 
sess = tf.Session()
filenames = glob.glob(imagenet_eval_dir + "/validation-*")
image, label, iterator = eval_input_fn(filenames, get_preprocess_fn())

IFX = ifx.IFX()
IFX.doInit(model_path)

# ================== warn up
for i in range(100):
  img = np.ones(3 * 224 * 224).astype('float32')
  time, output = IFX.doInference(img)

# ================== eval
time_cpp = []
time_python = []
correct = 0
loops = 1
while True:
  try:
    img, l = sess.run([image, label])
    img = img[0].transpose([2, 0, 1]).copy().astype('float32').reshape(3*224*224)
    
    start_time = datetime.now()
    time, output = IFX.doInference(img)
    time_python.append(millis(start_time))
    time_cpp.append(time)

    preds = np.argsort(output)[-5:]
    labels = l[0][0]

    if labels in preds:
      correct = correct + 1

    if (loops) % 5000 == 0:
      print("-------------> after " + str(loops) + ' images')
      print("Top-5: " + str(correct / loops))
      print("Time cpp: " + str(sum(time_cpp) / loops))
      print("Time python: " + str(sum(time_python) / loops))
    
    loops = loops + 1
  except tf.errors.OutOfRangeError:
    break

IFX.doRelease()
