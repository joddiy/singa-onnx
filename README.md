# singa-onnx
singa to onnx

# mlp.py
 - create singa autograd computing graph and save onnx model

# read-onnx
- read onnx model and use caffe to implement it

# mlponnx2singa
- read onnx model and create singa computing graph

# support layers
- LeakyRelu
- Softmax
- AddBias
- Add
- MatMul
- Flatten
- since onnx have not support save optimizer so we cannot save singa optimizer

# autograd.py
- computing graph and the code to save onnx model

# singa verion
- singa.__version__:1101
- installed from conda