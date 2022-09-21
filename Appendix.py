from grpc.beta import implementations
import numpy as np
import tensorflow as tf
import time
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.examples.tutorials.mnist import input_data

# reading data
def client_pre():
        batch_size = 1
        mnist = input_data.read_data_sets('../Desktop/tensorflow_test/MNIST_data/',one_hot=True)
        data,batch_y0 = mnist.train.next_batch(batch_size)
        
# Create stubs and build connections using IP addresses and port numbers
channel = implementations.insecure_channel('219.224.25.11',9000)
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

# Create a request
# initialize a request
request = predict_pb2.PredictRequest()
request.model_spec.name = 'nxj_test'
request.model_spec.signature_name = 'test_signature'        
request.inputs['input_x'].CopyFrom(tf.contrib.util.make_tensor_proto(data, shape=data.shape))

# Send the request and receive the return result
# predict
start_time = time.time()
result = stub.Predict(request)

# Some processing of the data; calculating the prediction time
print("cost time: {}".format(time.time()-start_time))
result_dict = {}
for key in result.outputs:
    tensor_proto = result.outputs[key]
    nd_array = tf.contrib.util.make_ndarray(tensor_proto)
    result_dict[key] = nd_array
y = []
m = []
for i in range(batch_size):
    y.append(np.where(batch_y0[i] == 1)[0][0])
for k in range(batch_size):
    if y[k] == result_dict['output'][k]:
        m.append('Right')
    if y[k] != result_dict['output'][k]:
        m.append('Wrong')
return result_dict,y,m

# Specify main functions and so on
def main():
        result,y,m = client_pre()
        print("-------原始结果-------")
        print(y)
        print("-------预测结果-------")
        print(list(result['output']))
        print(m)

if __name__ == '__main__':
        main()
