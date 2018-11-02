from singa import tensor
from singa.tensor import Tensor
from singa import autograd
from singa import optimizer
import numpy as np
import onnx
from onnx import numpy_helper

#import caffe2.python.onnx.backend as backend
import pickle
autograd.training = True

# prepare training data in numpy array

# generate the boundary
f = lambda x: (5 * x + 1)
bd_x = np.linspace(-1., 1, 2)
bd_y = f(bd_x)
# generate the training data
#x = np.random.uniform(-1, 1, 4)
x = np.array([0,0.5,-0.5,0.1])
#print(x)
y = f(x)# + 2 * np.random.randn(len(x))
#print(y)
# convert training data to 2d space
label = np.asarray([5 * a + 1 > b for (a, b) in zip(x, y)])
data = np.array([[a, b] for (a, b) in zip(x, y)], dtype=np.float32)

def to_categorical(y, num_classes):
    '''
    Converts a class vector (integers) to binary class matrix.
    Args
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    Return
        A binary matrix representation of the input.
    '''
    y = np.array(y, dtype='int')
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

label = to_categorical(label, 2).astype(np.float32)
print('train_data_shape:', data.shape)
print('train_label_shape:', label.shape)

inputs = Tensor(data=data)
target = Tensor(data=label)

with open('singonnx.pkl', 'rb') as f:
    model = pickle.load(f)
#with open('tensor.pkl', 'rb') as input:
#    a = pickle.load(input)
a={}
a['X'] = inputs
for i in model.graph.node:
    if(i.op_type == 'Constant'):
        a[str(i.output[0])]=tensor.from_numpy(numpy_helper.to_array(i.attribute[0].t))
        a[str(i.output[0])].stores_grad = True

#print(a)
sgd = optimizer.SGD(0.00)

# training process
for epoch in range(1):
    #print('auto grad x', tensor.to_numpy(Tensor(data=inputs.data, device=inputs.data.device)))
    for i in model.graph.node:
        if (i.op_type == 'Constant'):
            pass
            # do nothing
        if (i.op_type == 'LeakyRelu'):
            a[str(i.output[0])] = autograd.relu(a[str(i.input[0])])
        elif (i.op_type == 'Softmax'):
            a[str(i.output[0])] = autograd.soft_max(a[str(i.input[0])])
        elif (i.op_type == 'Add'):
            if(str(i.input[1])[-1] == 'b'):
                a[str(i.output[0])] = autograd.add_bias(a[str(i.input[0])], a[str(i.input[1])])
            else:
                a[str(i.output[0])] = autograd.add(a[str(i.input[0])],a[str(i.input[1])])
        elif (i.op_type == 'MatMul'):
            a[str(i.output[0])] = autograd.matmul(a[str(i.input[0])], a[str(i.input[1])])

    loss = autograd.cross_entropy(a['Y'], target)
    gradient,_ = autograd.backward(loss)
    for p, gp in gradient.items():
        #print(p.shape)
        #print(gp.shape)
        gp.reshape(p.shape)
        #print()
        #gp = gp.reshape(p.shape)
        #print(gp.shape)
        sgd.apply(0, gp, p, '')
    if (epoch % 100 == 0):
        print('training loss = ', tensor.to_numpy(loss)[0])

    #with open('onnxmodel.pkl', 'wb') as output:
    #    pickle.dump(model,output)
    #rep = backend.prepare(model, device="CPU")  # or "CPU"
    #outputs = rep.run(np.ones((2, 2)).astype(np.float32))
    #print(outputs[0])