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


#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

#include "OpenCL.h"
#include "OpenCLFile.h"



using namespace cl;


namespace opencl_private {


  int argumentCounter;
  Context context;
  CommandQueue queue;
  std::vector<Device> devices;
  std::vector<Event> events;
  std::map<const void*, Buffer*> buffers;
  std::map<const char*, OpenCLFile*> files;
  Kernel *kernel;
  std::map<cl_event, const char*> eventNames;

  Platform& getPlatformWithType(cl_device_type type, 
				std::vector<Platform>& platforms) {

    unsigned int i = 0;
    for (; i < platforms.size(); i++) {
      try {
	std::vector<Device> devices;
	platforms[i].getDevices(type, &devices);
	break;
      }
      catch (Error &e) {
      }
    }
    if (i >= platforms.size()) 
      throw Error(NO_PLATFORM_FOR_TYPE, "No platform found for type");

    return platforms[i];
  }



  void createContext(cl_device_type deviceType) {

    std::vector<Platform> platforms;
    Platform::get(&platforms);

    Platform& platform = getPlatformWithType(deviceType, platforms);

    cl_context_properties cprops[3] = 
      {CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(), 0};

    context = Context(deviceType, cprops);
  }


  Kernel *getKernel(const char* nameFileKernel) {
    if (!kernel) {
      std::map<const char*, OpenCLFile*>::iterator it;

      for (it = files.begin(); it != files.end(); ++it) {
	kernel = it->second->getKernel(nameFileKernel);
      }
    }

    return kernel;
  }



  void setArgOther(const char *kernelName, const void *arg) {

    Kernel *kernel = getKernel(kernelName);
    Buffer *buffer = buffers[arg];
    kernel->setArg(argumentCounter, *buffer);
    argumentCounter++;
  }

}

  
namespace opencl {


  
  void compile(const char *nameKernelFile, 
	       std::vector<std::string> &macros,
	       cl_device_type deviceType) {
    
    createContext(deviceType);
    devices = context.getInfo<CL_CONTEXT_DEVICES>();
    queue = CommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

    if (files.find(nameKernelFile) == files.end()) {
      files[nameKernelFile] =
	new OpenCLFile(nameKernelFile, macros, context);
    }
  }

  void launch(const char *kernelName, const NDRange& global,
	      const NDRange& local) {

    Kernel *kernel = getKernel(kernelName);
  
    //T *hostPointer = (T *) b->getInfo<CL_MEM_HOST_PTR>();
    //std::cout << kernel->getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;

    Event event;
    queue.enqueueNDRangeKernel(*kernel, cl::NullRange, 
			       global, local, &events, &event);
    events.push_back(event);
    eventNames[event()] = "launch";

    opencl_private::kernel = NULL;
    opencl_private::argumentCounter = 0;

    /*
      cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
      cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

      std::cerr << end - start << std::endl;
    */
  }


  void printEvent(Event& event) {
    cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

    const char* s = eventNames[event()] ? eventNames[event()] : "unknown";

    std::cerr << s <<  ": " << "start, " << start
	      << ", end: " << end
	      << ", total: " << end - start
	      << std::endl;
  }

  void printEventInfo() {

     for (unsigned int i = 0; i < events.size(); i++) {
       Event& e = events[i];
       printEvent(e);
     }
  }

  void deleteBuffers() {
    std::map<const void*, Buffer*>::iterator it;

    for (it = buffers.begin(); it != buffers.end(); ++it) {
      Buffer *b = it->second;
      delete b;
    }

    buffers.clear();
  }


  void sync() {
    Event::waitForEvents(events);
    printEventInfo();
    deleteBuffers();
    events.clear();
  }


  
  const char* resolveErrorCode(int error) {
    switch(error) {
    case CL_DEVICE_NOT_FOUND:                       
      return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:                   
      return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:                 
      return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:          
      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:                       
      return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:                     
      return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE:           
      return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP:                       
      return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH:                  
      return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:             
      return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE:                  
      return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE:                            
      return "CL_MAP_FAILURE";
    case CL_INVALID_VALUE:                          
      return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:                    
      return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:                       
      return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:                         
      return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:                        
      return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:               
      return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE:                  
      return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:                       
      return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:                     
      return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:        
      return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE:                     
      return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER:                        
      return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY:                         
      return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:                  
      return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:                        
      return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:             
      return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:                    
      return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION:              
      return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL:                         
      return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:                      
      return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:                      
      return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:                       
      return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:                    
      return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:                 
      return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:                
      return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:                 
      return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:                  
      return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:                
      return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:                          
      return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:                      
      return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT:                      
      return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE:                    
      return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL:                      
      return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE:               
      return "CL_INVALID_GLOBAL_WORK_SIZE";
    case CL_INVALID_PROPERTY:                       
      return "CL_INVALID_PROPERTY";
    case CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR:    
      return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case CL_PLATFORM_NOT_FOUND_KHR:                 
      return "CL_PLATFORM_NOT_FOUND_KHR";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:             
      return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:  
      return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case FILE_NOT_FOUND:
      return "FILE_NOT_FOUND";
    case NO_PLATFORM_FOR_TYPE:
      return "NO_PLATFORM_FOR_TYPE";
    default: 
      return "unknown";
    }
  }
}
