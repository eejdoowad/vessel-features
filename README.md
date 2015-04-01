Ridge-Features
==============

Status: In Progress

A port of Matlab vessel feature detection algorithms to c++ with OpenCV.

This port is based on code provided by Binxiong Lin. 
http://www.cse.usf.edu/~bingxiong/research_topic1.html

The original Matlab algorithms were written by D. Kroon of the University of Twente in 2009.
http://www.mathworks.com/matlabcentral/fileexchange/24409-hessian-based-frangi-vesselness-filter

To setup:

1. Install OpenCV

  a. If using a Unix-based system, I suggest you use a package manager (homebrew on Mac or apt on Ubuntu)
  
  b. If using Windows:
    
    1. Download OpenCV version 2.4.11 for windows and unzip it. 
    
      Note: You can use a different version if you know modify the .props files (opencv_2411_debug.props and opencv_2411_release.props)
    
    2. Make a new environment variable named 'OPENCV_DIR' with value 'path/to/OpenCV/Root/Directory'
    
    3. Add '%OPENCV_DIR%\build\x86\vc12\bin' to your Path environment variable
    
      Note: You can use a different version of Visual Studio if you modifying 'vc12' in the path above to some other version
        vc12: Visual Studio 2013
        vc11: Visual Studio 2012
        vc10: Visual Studio 2010

2. Clone this repository

3. Build the program (an executable called vessel-features should be generated in the bin directory)

  a. If using a Unix-based system like Ubuntu or Mac OSX, run make in build/unix
  
  b. If using Windows, open the solution file and hit debug

4. Run the generated executable and see if it works
