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

import numpy as np
#import caffe2.python.onnx.backend as backend
import pickle
autograd.training = True

data = np.ones((2,3,4,4),dtype=np.float32)
label = np.array([[0.0,1.0],[1.0,0.0]],dtype=np.float32)

print('train_data_shape:', data.shape)
print('train_label_shape:', label.shape)

inputs = Tensor(data=data)
target = Tensor(data=label)

#conv = autograd.Conv2d(3, 4, 3, padding=1, bias=True)
#pooling = autograd.MaxPool2d(2, 2, padding=0)
linear = autograd.Linear(32 * 28 * 28, 2)



sgd = optimizer.SGD(0.00)

# training process
for i in range(1):
    #x = conv(inputs)
    #x = autograd.max_pool_2d(inputs,2, 2, padding=0)
    x = autograd.flatten(inputs)
    x = linear(x)
    x = autograd.soft_max(x)
    loss = autograd.cross_entropy(x, target)
    gradient,model = autograd.backward(loss)
    for p, gp in gradient.items():
        gp.reshape(p.shape)
        #print()
        #gp = gp.reshape(p.shape)
        #print(gp.shape)
        sgd.apply(0, gp, p, '')
    if (i % 100 == 0):
        print('training loss = ', tensor.to_numpy(loss)[0])

    #with open('singonnx.pkl', 'wb') as output:
    #    pickle.dump(model,output)
    #rep = backend.prepare(model, device="CPU")  # or "CPU"
    #outputs = rep.run(np.ones((2, 2)).astype(np.float32))
    #print(outputs[0])