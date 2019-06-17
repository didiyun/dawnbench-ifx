import ifx
import numpy as np
import os
from optparse import OptionParser

parser = OptionParser() 
parser.add_option("-m", "--model", action="store", 
                  dest="model", 
                  help="input model file") 
(options, args) = parser.parse_args()

model_path = os.path.abspath(options.model)

# init model
IFX = ifx.IFX()
IFX.doInit(model_path)

# warn up
for i in range(100):
  input = np.ones([3 * 224 * 224]).astype('float32')
  time, output = IFX.doInference(input)

# benchmark
for i in range(10):
  input = np.ones([3 * 224 * 224]).astype('float32')
  time_ms, output = IFX.doInference(input)
  print(time_ms)

IFX.doRelease()
