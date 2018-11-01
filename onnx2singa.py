from singa import tensor
from singa.tensor import Tensor
from singa import autograd
from singa import optimizer
import numpy as np

#import caffe2.python.onnx.backend as backend
import pickle
autograd.training = True

# prepare training data in numpy array

# generate the boundary
f = lambda x: (5 * x + 1)
bd_x = np.linspace(-1., 1, 2)
bd_y = f(bd_x)
# generate the training data
x = np.random.uniform(-1, 1, 4)
y = f(x) + 2 * np.random.randn(len(x))
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

with open('onnxmodel.pkl', 'rb') as input:
    model = pickle.load(input)
with open('tensor.pkl', 'rb') as input:
    a = pickle.load(input)
    print(a)
for key,value in a.item():
    a[key] = tensor.from_numpy(a[key])
    a[key].stores_grad =True
print(a)
sgd = optimizer.SGD(0.05)

# training process
for i in range(1):
    #print('auto grad x', tensor.to_numpy(Tensor(data=inputs.data, device=inputs.data.device)))
    x = autograd.matmul(inputs, w0)
    x = autograd.add_bias(x, b0)
    # x = autograd.relu(x)
    x2 = autograd.matmul(x, w2)
    x2 = autograd.add_bias(x2, b2)
    x1 = autograd.matmul(x, w1)
    x1 = autograd.add_bias(x1, b1)
    x3 = autograd.add(x1, x2)
    #print('auto grad x',tensor.to_numpy(x))
    #print('auto grad x',x)
    #print('auto grad x', tensor.to_numpy(Tensor(data=x.data,device=x.data.device)))
    #print('---auto end---')
    #x = autograd.matmul(x, w1)
    #x = autograd.add_bias(x, b1)
    x3 = autograd.soft_max(x3)
    loss = autograd.cross_entropy(x3, target)
    #print(autograd.backward(loss))
    gradient,model = autograd.backward(loss)
    for p, gp in gradient.items():
        #print(p.shape)
        #print(gp.shape)
        gp.reshape(p.shape)
        #print()
        #gp = gp.reshape(p.shape)
        #print(gp.shape)
        sgd.apply(0, gp, p, '')
    if (i % 100 == 0):
        print('training loss = ', tensor.to_numpy(loss)[0])

    with open('onnxmodel.pkl', 'wb') as output:
        pickle.dump(model,output)
    #rep = backend.prepare(model, device="CPU")  # or "CPU"
    #outputs = rep.run(np.ones((2, 2)).astype(np.float32))
    #print(outputs[0])