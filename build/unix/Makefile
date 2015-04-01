# Makefile

CC=g++
CFLAGS=-g -Wall -Wextra -std=c++11 -I/usr/include
LDFLAGS=-L/usr/local/lib/ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_stitching

ROOTDIR=../../
OUTDIR=$(ROOTDIR)bin/
EXECUTABLE=$(OUTDIR)vessel_features
OBJECTDIR=$(OUTDIR)objects/
SOURCEDIR=$(ROOTDIR)source/

SOURCES=$(wildcard $(SOURCEDIR)*.cpp)
SOURCES_NO_PATH=$(patsubst $(SOURCEDIR)%,%,$(SOURCES))
# Mapping: SOURCES => SOURCES_NO_PATH
# Example: ../../sources/main.cpp => main.cpp
HEADERS=$(wildcard $(SOURCEDIR)*.hpp)
OBJECTS=$(patsubst %.cpp,$(OBJECTDIR)%.o,$(SOURCES_NO_PATH))


all: $(OBJECTDIR) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) $(HEADERS)
	$(CC) -o $@ $(OBJECTS) $(LDFLAGS)

$(OBJECTDIR):
	mkdir -p $(OBJECTDIR)

$(OBJECTDIR)%.o: $(SOURCEDIR)%.cpp
	$(CC) -c -o $@ $< $(CFLAGS)

clean:
	rm $(EXECUTABLE)
	rm -rf $(OBJECTDIR)

.PHONY: all clean

# End Makefile