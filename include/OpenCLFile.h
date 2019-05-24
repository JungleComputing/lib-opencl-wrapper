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


#ifndef OPENCL_FILE_H
#define OPENCL_FILE_H



#define __CL_ENABLE_EXCEPTIONS
#include <vector>
//#define __NO_STD_VECTOR //Disable std::vector 
#include <CL/cl.hpp>



using namespace cl;



class OpenCLFile {
    private:
	const char* fileName;
	Context context;
	std::vector<Device> devices; 
	std::string sourceString;
	Program::Sources source;
	Program program;

        void getSource(std::string &source, const char *kernelName);
        void build(Program &program, std::vector<Device> &devices);



    public:
	OpenCLFile(const char* fileName, std::vector<std::string> &macros, 
		   Context& context);
	~OpenCLFile(); 

       void build(const char *params);
        void saveBinaries();
	void printBuildInfo(); 
	Kernel *getKernel(const char *kernelName);
	Device& getDevice();
	void insertMacros(std::vector<std::string> &macros);
};



#endif
