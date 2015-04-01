# Makefile

CC=g++
CFLAGS=-g -Wall -Wextra -std=c++11 -I/usr/local/include/opencv -I/usr/local/include/opencv2
LDFLAGS=-L/usr/local/lib/ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_stitching

ROOTDIR=../../
OUTDIR=$(ROOTDIR)bin/
EXECUTABLE=$(OUTDIR)vessel_features
SOURCEDIR=$(ROOTDIR)source/

SOURCES=$(wildcard $(SOURCEDIR)*.cpp)
SOURCES_NO_PATH=$(patsubst $(SOURCEDIR)%,%,$(SOURCES))
# Mapping: SOURCES => SOURCES_NO_PATH
# Example: ../../sources/main.cpp => main.cpp
HEADERS=$(wildcard $(SOURCEDIR)*.hpp)
OBJECTS=$(patsubst %.cpp,$(OUTDIR)%.o,$(SOURCES_NO_PATH))

$(OUTDIR)%.o: $(SOURCEDIR)%.cpp $(HEADERS)
	$(CC) -c -o $@ $< $(CFLAGS)

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) $(HEADERS)
	$(CC) -o $@ $(OBJECTS) $(LDFLAGS)

clean:
	rm $(EXECUTABLE)
	rm $(OBJECTS)

.PHONY: all clean

# End Makefile