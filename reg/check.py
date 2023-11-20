import numpy
import onnxruntime as rt

X_train=numpy.load('X_train.npy')
for model_name in ['RF','KNN','LR']:
    print(model_name)
    session = rt.InferenceSession(f'{model_name}.onnx', providers=["CPUExecutionProvider"])
    y_train_pred=numpy.load(f'{model_name}_predict.npy')
    y_pred_onnx=session.run(None, {"X": X_train.astype(numpy.float32)})[0].ravel()
    # возвращает ответ в виде [arr] поэтому необходимо указать [0]
    # arr имеет shape (N,1) поэтому необходимо еще вытягивать (метод ravel())
    # если конвертер определял как float32 тогда и данные подавай в float32 в противном случае вернет 
    # ошибку о некорректности типов данных
    delta=numpy.abs((y_train_pred-y_pred_onnx)/y_train_pred).max()
    print('delta:',delta)
    print()