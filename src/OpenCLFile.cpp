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
#include <sstream>
#include <string>
#include <cstdlib>

#include "OpenCLFile.h"
#include "OpenCL.h"



using namespace cl;



OpenCLFile::OpenCLFile(const char *fileName, std::vector<std::string> &macros, 
		       Context& context) :

  fileName(fileName) {

  getSource(sourceString, fileName);

  devices = context.getInfo<CL_CONTEXT_DEVICES>();

  insertMacros(macros);

  //std::cerr << sourceString << std::endl;;

  source.push_back(std::make_pair(sourceString.c_str(),
				  sourceString.length()+1));
  
  program = Program(context, source);
  
  build(std::getenv("OCL_OPTIONS"));
}


OpenCLFile::~OpenCLFile() {
}


void OpenCLFile::insertMacros(std::vector<std::string> &macros) {
    for (unsigned i = 0; i < macros.size(); i++) {
	std::string macro = "#define ";
	macro += macros[i];
	macro += "\n";
	sourceString.insert(0, macro);
    }
}




void OpenCLFile::getSource(std::string &sourceString, 
			   const char *fileName) {

  std::string fileNameWithExt(fileName);
  fileNameWithExt += ".cl";

  std::ifstream file(fileNameWithExt.c_str());
  if (!file.is_open()) {
    throw Error(FILE_NOT_FOUND, fileNameWithExt.c_str());
				  
  }

  sourceString = std::string(std::istreambuf_iterator<char>(file),
			     (std::istreambuf_iterator<char>()));
}


void OpenCLFile::build(const char *params) {
    try {
	program.build(devices, params);
    }
    catch (Error &err) {
	printBuildInfo();
	throw err;
    }
    printBuildInfo();
}


void OpenCLFile::printBuildInfo() {
    for (unsigned int i = 0; i < devices.size(); i++) {
	std::cout << "Build Status: " << 
	    program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[i]) << 
	    std::endl;
	std::cout << "Build Options: " << 
	    program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices[i]) << 
	    std::endl;
	std::cout << "Build Log:\n" << 
	    program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[i]) << 
	    std::endl;
    }
}


Kernel *OpenCLFile::getKernel(const char *kernelName) {
    return new Kernel(program, kernelName);
}


Device& OpenCLFile::getDevice() {
    return devices[0];
}


void OpenCLFile::saveBinaries() {
    cl_uint nrDevices;

    nrDevices = program.getInfo<CL_PROGRAM_NUM_DEVICES>();

    std::size_t binarySizes[nrDevices];

    program.getInfo(CL_PROGRAM_BINARY_SIZES, binarySizes);

    unsigned char* binaries[nrDevices];

    for (unsigned int i = 0; i < nrDevices; i++) {
	binaries[i] = new unsigned char[binarySizes[i] + 1];
    }
    program.getInfo(CL_PROGRAM_BINARIES, binaries);

    std::string outputFileName(fileName);
    outputFileName = outputFileName.substr(0, outputFileName.length() - 3);
    outputFileName += ".ptx";

    if (nrDevices == 1) {
	std::ofstream out(outputFileName.c_str());
	out << binaries[0];
	out.close();
    }
    else {
	for (unsigned int i = 0; i < nrDevices; i++) {
	    std::ostringstream oss(outputFileName.c_str());
	    oss << "_" << i; 
	    std::ofstream out(oss.str().c_str());
	    out << binaries[i];
	    out.close();
	}
    }

    for (unsigned int i = 0; i < nrDevices; i++) {
	delete [] binaries[i]; 
    }
}


void addMacro(std::vector<std::string> &macros, std::string s, int v) {
    std::ostringstream convert;   
    convert << " " << v;    
    s += convert.str(); 
    macros.push_back(s);
}
