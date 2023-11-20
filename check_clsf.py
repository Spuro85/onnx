import onnxruntime as rt
import numpy

X_test=numpy.load('X_test.npy')
for model_name in ['xgb','CatB','lgb']:
    sess = rt.InferenceSession(f"{model_name}.onnx", providers=["CPUExecutionProvider"])
    y_test_pred_onnx=numpy.array([row[2] for row in sess.run(output_names=[],input_feed={sess.get_inputs()[0].name:X_test})[1]])
    y_test_pred=numpy.array([row[2] for row in numpy.load(f'{model_name}_predict.npy')])
    print(model_name)
    print(numpy.abs((y_test_pred-y_test_pred_onnx)/y_test_pred).max())
    print('')