#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#



from singa import tensor
from singa.tensor import Tensor
from singa import autograd
from singa import optimizer
from singa import *
from singa import sonnx
import onnx
from singa import device
import numpy as np
autograd.training = True
np.random.seed(0)
data = np.random.randn(4,2).astype(np.float32)
h = np.random.randn(4,2).astype(np.float32)
label = np.random.randn(4,2).astype(np.float32)
print('train_data_shape:', data.shape)
print('train_label_shape:', label.shape)


rnn = autograd.RNN(2,2)



sgd = optimizer.SGD(0.00)
dev = device.get_default_device()
# training process
for i in range(1):
    inputs = tensor.Tensor(device=dev, data=data, stores_grad=False)
    h0 = tensor.Tensor(device=dev, data=h, stores_grad=False)
    targets = tensor.Tensor(device=dev, data=label, requires_grad=False, stores_grad=False)
    x = rnn(inputs,h0)
    loss = autograd.mse_loss(x, targets)
    gradient = autograd.backward(loss)
    for p, gp in gradient:
        sgd.apply(0, gp, p, '')
    if (i % 100 == 0):
        print('training loss = ', tensor.to_numpy(loss)[0])


model=sonnx.get_onnx_model(loss,inputs,target)

onnx.save(model, 'linear.onnx')

