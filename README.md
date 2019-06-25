# DiDi Cloud IFX

IFX is an DiDi Cloud self-research AI Accelerated Inference Engine. It Provide low latency and high throughput neural network acceleration solutions. Neural network models based on tensorflow, pytorch, caffe and other frameworks can be deployed to heterogeneous devices such as NVIDIA GPU, ARM, etc. At present, IFX has been widely used in face detection, OCR, speech recognition, ETA and other businesses.

滴滴云自研的AI加速推理引擎 IFX，提供低延时，高吞吐的神经网络实现方案，支持 tensorflow，pytorch，caffe 等神经网络模型一键部署到 NVIDIA GPU，ARM 等异构设备。目前该引擎已经广泛应用于滴滴内部人脸检测、OCR、语音识别、eta 等各项业务。

## Enviroment Prepare

- **Host**：DiDi Cloud General T4 instance with 8 vCPUs，1 Nvidia Tesla T4，16GB GPU Memory，16GB CPU Memory
- **Software**: IFX, CUDA-10.0, CUDNN-7.5.0, Tensorflow-1.13.1, TensorRT-5.1.2.2
- **Model**: [tensorflow slim resnet_v1_50](https://github.com/tensorflow/models/tree/master/research/slim)


```bash
pip install tensorflow==1.13.1
```

## How To Run?

```bash
# Benchmark with fake input
MODEL_DIR=`pwd`/resnet_v1_50_int8.ifxmodel
python benchmark.py --model $MODEL_DIR

# Benchmark with ImageNet evaluation tfrecord
MODEL_DIR=`pwd`/resnet_v1_50_int8.ifxmodel
DATA_DIR="ImageNet evaluation tfrecord directory"
python eval.py --model $MODEL_DIR  --data_dir $DATA_DIR
```

Note: Get DiDi Cloud General T4 instance here: [https://www.didiyun.com/production/gpu.html](https://www.didiyun.com/production/gpu.html)