1. [下载tensorflow源码](https://github.com/tensorflow/tensorflow)

2. 拷贝代码文件夹`C++`到`tensorflow/lite/tools/make/`

3. 在`Makefile`文件中增加如下代码

   ```
   HELLOW_TFLIET := hellow_tf
   HELLOW_TFLIET_BINARY := $(BINDIR)$(HELLOW_TFLIET)
   
   HELLOW_TFLIET_SRCS := \
   tensorflow/lite/tools/make/C++/model.cc \
   tensorflow/lite/tools/make/C++/example.cc
   
   INCLUDES += \
   -Itensorflow/lite/tools/make/C++/
   	
   ALL_SRCS += \
     $(HELLOW_TFLIET_SRCS)
   
   CORE_CC_EXCLUDE_SRCS += \
   $(wildcard tensorflow/lite/tools/make/C++/model.cc) \
   $(wildcard tensorflow/lite/tools/make/C++/example.cc)
   
   HELLOW_TFLIET_OBJS := $(addprefix $(OBJDIR), \
   $(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(HELLOW_TFLIET_SRCS))))
   
   $(HELLOW_TFLIET): $(LIB_PATH) $(HELLOW_TFLIET_OBJS) 
   	@mkdir -p $(BINDIR)
   	$(CXX) $(CXXFLAGS) $(INCLUDES) \
   	-o $(HELLOW_TFLIET_BINARY) $(HELLOW_TFLIET_OBJS) \
   	$(LIBFLAGS) $(LIB_PATH) $(LDFLAGS) $(LIBS)
   ```
4. 执行编译命令`make hellow_tf -j8 -f tensorflow/lite/tools/make/Makefile`