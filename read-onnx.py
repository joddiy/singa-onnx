import pickle
import numpy as np
import caffe2.python.onnx.backend as backend
with open('onnxmodel.pkl', 'rb') as input:
    model = pickle.load(input)
    rep = backend.prepare(model, device="CPU")  # or "CPU"
    outputs = rep.run(np.ones((4, 2)).astype(np.float32))
    print(outputs[0])