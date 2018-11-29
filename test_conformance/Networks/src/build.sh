#!/bin/bash 
# Arg 1: Build type: Release/Debug
# Arg 2: Architecture: x86/x64

if [ "$#" -eq 0 ] || [ "$#" -eq 1 ]
then
echo "Invalid number of parameters. First must be \"Debug\" or \"Release\". Second must be \"x86\" or \"x64\""
exit 1
fi

buildType=$1
arch=$2

if [ "$arch" != "x86" ] && [ "$arch" != "x64" ]
then
echo "Architecture $arch is not supported"
exit 1
fi

if [ "$buildType" != "Release" ] && [ "$buildType" != "Debug" ]
then
echo "Build Tape $buildType is not supported"
exit 1
fi

# Check environment requirements
command -v cmake >/dev/null 2>&1 || { echo "CMake is not installed!" >&2; exit 1; }

if [ -z "$OPENVX_FOLDER" ]
then
echo "OPENVX_FOLDER environment variable not set. Make sure set it to the location of your OpenVX installation folder. E.g. \"export OPENVX_FOLDER=/home/openvx/x64/Release\""
exit 1
fi

# Start build

mkdir -p _build
cd _build
mkdir -p $arch
cd $arch
mkdir -p $buildType
cd $buildType

echo "Starting build... $arch $buildType"

cmake -DCMAKE_BUILD_TYPE=$buildType -DBUILD_ARCH=$arch ../../.. 
cmake --build .

