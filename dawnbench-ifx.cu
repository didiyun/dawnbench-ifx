#include <chrono>
#include <iostream>
#include <fstream>

#include <cuda_runtime.h>
#include <cuda.h>

#include <opencv.hpp>

#include "glog/logging.h"
#include "framework/graph.h"

using namespace std;

int cal_topk(const vector<float>& cls_scores, int topk, vector<int>& topn) {
  int size = cls_scores.size();
  std::vector< std::pair<float, int> > vec;
  vec.resize(size);
  for (int i = 0; i < size; i++) {
    vec[i] = std::make_pair(cls_scores[i], i);
  }

  std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                    std::greater< std::pair<float, int> >());

  for (int i = 0; i < topk; i++) {
    float score = vec[i].first;
    int index = vec[i].second;
    topn.push_back(index);
  }
  return 0;
}

void splits(const string& s, vector<string>& tokens, const string& delimiters = " ") {
  string::size_type lastPos = s.find_first_not_of(delimiters, 0);
  string::size_type pos = s.find_first_of(delimiters, lastPos);
  int i = 0;
  while (i < 2) {
  //while (string::npos != pos || string::npos != lastPos) {
    tokens.push_back(s.substr(lastPos, pos - lastPos));
    lastPos = s.find_first_not_of(delimiters, pos);
    i++;
  }
}

int main(int argc, char **argv) {
  // init
  cuInit(0);

  // argvs
  const int benchmark = atoi(argv[1]);  // 0: imagenet validation, 1: benchmark
  const string model_file = argv[2];

  // load ifx models & create session
  IFX::Graph graph;
  bool ret = graph.load_model(model_file.c_str());
  if (!ret) {
    LOG(ERROR) << "Init ifx model failed.";
    return -1;
  }
  IFX::Session sess = graph.create_session();

  int64_t time_cost = 0;
  IFX::Tensor *output = new IFX::Tensor();
  IFX::Tensor *input = new IFX::Tensor(1, 3, 224, 224, "NCHW", 4);
  // do benchmark
  if (benchmark == 1) {
    LOG(INFO) << "~~~ start benchmark ~~~";
    // step1: warn up
    for (int i = 0; i < 1000; i++) {
      sess.feed_back("input", input);
      time_cost = sess.run("output", output);
    }

    // step2: benchmark
    int loop = 10000;
    int64_t total_time = 0, min = 1000000, max = -1;
    for (int i = 0; i < loop; i++) {
      sess.feed_back("input", input);
      time_cost = sess.run("output", output);
      total_time += time_cost;
      if (time_cost > max) max = time_cost;
      if (time_cost < min) min = time_cost;
    }
    float avg = 1.f * total_time / loop;
    float qps = 1000.0f * 1000.f / avg;
    LOG(INFO) << "iters: " << loop;
    LOG(INFO) << "time (avg, min, max) us: " << avg << ", " << min << ", " << max;
    LOG(INFO) << "qps: " << qps;
    return 0;
  }

  // do imagenet validation
  const string label_file = argv[3];
  const string validate_image_dir = argv[4];

  // prepare labels & images
  LOG(INFO) << "~~~ start imagenet validation ~~~";
  ifstream label_stream(label_file.c_str());
  vector<string> labels_string;
  vector<string> images;
  vector<int> labels_index;

  string line;
  while (getline(label_stream, line)) {
    labels_string.push_back(string(line));
  }

  for(int i = 0; i < labels_string.size(); i++) {
    vector<string> split_label;
    splits(labels_string[i], split_label);
    images.push_back(split_label[0]);
    labels_index.push_back(atoi(split_label[1].c_str()));
  }

  vector<float> cls_scores;
  cls_scores.resize(1024);
  int count_accurcy_top5 = 0;
  for (int i = 0; i < images.size(); i++) {
    // do preprocessing
    cv::Mat image_rgb, image_resize, image_crop, image_norm;
    cv::Mat image = cv::imread(string(validate_image_dir + "/"+ images[i]));
    cv::cvtColor(image, image_rgb, CV_BGR2RGB);
    // resize & crop
    float short_side = std::min(image_rgb.cols, image_rgb.rows);
    float scale_resize = 256.f / short_side;
    cv::Size size = cv::Size(image_rgb.cols * scale_resize, image_rgb.rows * scale_resize);
    cv::resize(image_rgb, image_resize, size, CV_INTER_CUBIC);
    const cv::Rect roi((size.width - 224 + 1) / 2, (size.height - 224 + 1) / 2, 224, 224);
    image_crop = image_resize(roi).clone();

    // normalize
    std::vector<float> mean_value{0.485, 0.456, 0.406};
    std::vector<float> std_value{0.229, 0.224, 0.225};
    std::vector<cv::Mat> rgbChannels(3);
    cv::split(image_crop, rgbChannels);
    for (auto ii = 0; ii < 3; ii++) {
      rgbChannels[ii].convertTo(rgbChannels[ii], CV_32FC1, 1.0 / std_value[i], (0.0 - mean_value[i]) / std_value[i]);
    }
    cv::merge(rgbChannels, image_norm);

    // opencv2ifx tensor
    memcpy(input->template mutable_data<float>(), image_norm.data, 3 * 224 * 224 * sizeof(float));

    // do inference
    sess.feed_back("input", input);
    time_cost = sess.run("Unit10_gemm", output);

    // do postprocessing
    memcpy(&cls_scores[0], output->template data<float>(), 1024 * sizeof(float));
    std::vector<int> top5;
    cal_topk(cls_scores, 5, top5);

    for(auto ii = 0; ii < top5.size(); ii++) {
      if(top5[ii] == labels_index[i]) {
        count_accurcy_top5++;
      }
    }
  }

  LOG(INFO) << "Images: " << images.size();
  LOG(INFO) << "Top5 scores: " << float(count_accurcy_top5) / images.size();
  
  return 0;
}