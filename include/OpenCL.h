/*
 * Copyright 2019 Vrije Universiteit Amsterdam, The Netherlands
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#ifndef OPENCL_H
#define OPENCL_H

#include <map>


#define __CL_ENABLE_EXCEPTIONS
//#define __NO_STD_VECTOR //Disable std::vector
#include <vector>
#include <CL/cl.hpp>

#include "OpenCLFile.h"


// extra error codes
#define FILE_NOT_FOUND -128
#define NO_PLATFORM_FOR_TYPE -129

namespace opencl_private {

  extern int argumentCounter;
  extern Context context;
  extern CommandQueue queue;
  /* vector<Device> devices; */
  extern std::vector<Event> events;
  extern std::map<const void*, Buffer*> buffers;
  extern std::map<cl_event, const char*> eventNames;
  /* std::map<const char*, OpenCLFile*> files; */
  /* Kernel *kernel; */


  Platform& getPlatformWithType(cl_device_type type, 
				std::vector<Platform>& platforms);

  void createContext(cl_device_type deviceType);

  Kernel *getKernel(const char* nameFileKernel);

  void setArgOther(const char *kernelName, const void *arg);
}



using namespace opencl_private;

namespace opencl {

  
  
  enum TYPE { READ_WRITE, READ };
  
  void compile(const char *nameKernelFile, 
	       std::vector<std::string> &macros,
	       cl_device_type deviceType);
  
  const char* resolveErrorCode(int error);

  
  template<typename T> inline void setArg(const char* kernelName, T arg) {
    setArgOther(kernelName, arg);
  }

  template<> inline void setArg<int>(const char* kernelName, int arg) {
    Kernel *kernel = getKernel(kernelName);
    kernel->setArg(opencl_private::argumentCounter, arg);
    argumentCounter++;
  }

  template<> inline void setArg<float>(const char* kernelName, float arg) {
    Kernel *kernel = getKernel(kernelName);
    kernel->setArg(opencl_private::argumentCounter, arg);
    argumentCounter++;
  }

  template<typename T> inline Buffer *allocate2(T *arg, TYPE t, ::size_t size) {
    cl_mem_flags flags;

    switch (t) {
    case READ:
      flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
      break;
    case READ_WRITE:
      flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
      break;
    }

    return new Buffer(context, flags, size, arg);
  }

  template<typename T> inline void allocate(T *arg, TYPE t, ::size_t size) {
    buffers[arg] = allocate2<T>(arg, t, size);
  }

  template<typename T> inline void transferToDevice(T *arg, TYPE t,
  						    ::size_t size) {
    Buffer *b = allocate2<T>(arg, t, size);
    Event event;
    queue.enqueueWriteBuffer(*b, CL_FALSE, 0, size, arg, &events, &event);
    events.push_back(event);
    eventNames[event()] = "transferTodevice";
    buffers[arg] = b;
  }


  /* template<typename T> inline void setArg(const char* kernelName, T arg) { */
  /*   Kernel *kernel = getKernel(kernelName); */
  /*   Buffer *buffer = buffers[arg]; */
  /*   //kernel->setArg(argumentCounter, *buffer); */
  /*   setArg2<T>(kernel, argumentCounter, buffer); */
  /*   argumentCounter++; */
  /* } */

  /*
  template<typename T> inline void setArg(const char* kernelName, T arg) {
    setArg(kernelName, arg, identity<T>());
  }
  */


  template<typename T> inline void transferFromDevice(T *arg) {
    Buffer *b = buffers[arg];
    unsigned int size = b->getInfo<CL_MEM_SIZE>();
    Event event;
    queue.enqueueReadBuffer(*b, CL_FALSE, 0, size, arg, &events, &event);
    eventNames[event()] = "transferFromDevice";
    events.push_back(event);
  }

  void launch(const char* kernelName, const NDRange& global, 
	      const NDRange& local);
  void sync();
	    

  template<typename T> inline void deallocate(T *arg) {
    /*
    Buffer *b = buffers[arg];
    delete b;
    buffers[arg] = NULL;
    */
  }
};



#endif
