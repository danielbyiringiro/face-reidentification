#!/bin/bash

# Check if the weights directory does not exist, then create it
if [ ! -d "weights" ]; then
  mkdir weights
fi

# Clean the weights directory
rm -rf weights/*.onnx

# Download the files and save them to the weights directory
curl -L -o weights/det_2.5g.onnx https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_2.5g.onnx
curl -L -o weights/det_500m.onnx https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_500m.onnx
curl -L -o weights/det_10g.onnx https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_10g.onnx
curl -L -o weights/w600k_mbf.onnx https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_mbf.onnx
curl -L -o weights/w600k_r50.onnx https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_r50.onnx
