CC = nvcc
CVLABS = $(shell pkg-config opencv --cflags --libs)

FLAGS = -std=c++11 -arch sm_75                            \
        -I/usr/local/cuda/include/                        \
        -I/usr/local/include/                             \
        -L/usr/local/cuda-10.0/targets/x86_64-linux/lib/  \
        -I/usr/include/opencv2/                           \
        -L/usr/local/lib/                                 \
        -lglog -lifx -lcuda -lcudart $(CVLABS)

OBJS = dawnbench-ifx

all: $(OBJS)

$(OBJS):
  $(CC) $(OBJS).cu -o $(OBJS) $(FLAGS)

clean:
  rm -f $(OBJS)
