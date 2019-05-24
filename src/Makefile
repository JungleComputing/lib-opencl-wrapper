SRC		= OpenCL.cpp OpenCLFile.cpp
HDR		= OpenCL.h OpenCLFile.h
LIBRARY		= libopencl-wrapper.a

INCLUDE_DIRS	= -I ../include

CC		= g++
CFLAGS		= -g -Wall $(INCLUDE_DIRS)

AR		= ar
AR_FLAGS	= rcs 

DEPFILE		= .depends
DEPFLAGS	= 

OBJ		= $(SRC:%.cpp=%.o)

.PHONY:		clean


$(LIBRARY): $(DEPFILE) $(OBJ) 
	$(AR) $(AR_FLAGS) $@ $(OBJ) 

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(DEPFILE):
	$(CC) -M $(DEPFLAGS) $(CFLAGS) $(SRC) > $@

clean: 
	rm -f *.s *.o $(LIBRARY) $(DEPFILE)

sinclude $(DEPFILE)
