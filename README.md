# DiDi Cloud IFX

IFX is an DiDi Cloud self-research AI Accelerated Inference Engine. It Provide low latency and high throughput neural network acceleration solutions. Neural network models based on tensorflow, pytorch, caffe and other frameworks can be deployed to heterogeneous devices such as NVIDIA GPU, ARM, etc. At present, IFX has been widely used in face detection, OCR, speech recognition, ETA and other businesses.

滴滴云自研的AI加速推理引擎 IFX，提供低延时，高吞吐的神经网络实现方案，支持 tensorflow，pytorch，caffe 等神经网络模型一键部署到 NVIDIA GPU，ARM 等异构设备。目前该引擎已经广泛应用于滴滴内部人脸检测、OCR、语音识别、eta 等各项业务。

## Enviroment Prepare

- **Host**: DiDi Cloud General T4 instance with 8 vCPUs，1 Nvidia Tesla T4，16GB GPU Memory，16GB CPU Memory
- **OS**: ubuntu 16.04
- **Software**: IFX, CUDA-10.0, OpenCV2, glog
- **Model**: Standard Resnet26

## How To Run?

```bash
# Benchmark with fake input
MODEL_DIR=`pwd`/resnet26.ifxmodel
./dawnbench-ifx 1 $MODEL_DIR

# Benchmark with ImageNet evaluation dataset
MODEL_DIR=`pwd`/resnet_v1_50_int8.ifxmodel
DATA_DIR="ImageNet evaluation dataset directory"
LABEL_FILE="ImageNet label file"
./dawnbench-ifx 0 $MODEL_DIR $LABEL_FILE $DATA_DIR
```

Note: Get DiDi Cloud General T4 instance here: [https://www.didiyun.com/production/gpu.html](https://www.didiyun.com/production/gpu.html)

**使用 AI 大师码【0220】购买滴滴云 GPU 享 9 折优惠!!!**