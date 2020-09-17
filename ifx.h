#pragma once

#include <vector>
#include <string>
#include <utility>
#include <map>
#include <list>
#include <memory>

using namespace std;

namespace IFX {

class Tensor;
class Context {
 public:
  Context();
};

class CPUContext : public Context {
 public:
  CPUContext();

  static CPUContext* context() {
    static CPUContext *ctx = new CPUContext();
    return ctx;
  }
};

class GPUContext : public Context {
 public:
  GPUContext();

  static GPUContext* context() {
    static GPUContext ctx;
    return &ctx;
  }
};


class Tensor {
 public:
  explicit Tensor(Context *ctx = CPUContext::context());
  
  Tensor(int n, int c, int h, int w, string format = "NCHW",
         int element_size = 4, Context *ctx = CPUContext::context());

  template <class T>
  inline const T* data() const {
    return static_cast<const T*>(data_);
  }

  template <class T>
  inline T* mutable_data() {
    return static_cast<T*>(data_);
  }

 private:
  void* data_;
};

class Session;
class Graph {
 public:
  Graph();
  ~Graph();

  Session create_session() const;
  bool load_model(const char* model);
  bool load_model(FILE *fp);
};

class Session {
 public:
  ~Session();
  
  bool feed_back(string name, Tensor *input);
  float run(string name, Tensor *output);

 protected:
  friend Session Graph::create_session() const;
  explicit Session(const Graph *graph);
};

}  // namespace IFX
