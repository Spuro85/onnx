import numpy

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from skl2onnx import to_onnx

print('import lib: ok!')

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
print('train/test split: ok!')
# Train classifiers
dct={'RF':RandomForestRegressor,
    'KNN':KNeighborsRegressor,
    'LR':LinearRegression}
numpy.save('X_train.npy',X_train)
numpy.save('y_train.npy',y_train)
for model_name in dct.keys():
    print(model_name)
    reg = dct[model_name]()
    reg.fit(X_train, y_train)
    print(f'{model_name} fit: ok!')
    y_pred_train=reg.predict(X_train)
    print('r2:',round(r2_score(y_true=y_train,y_pred=y_pred_train),2))

    model_onx = to_onnx(reg, X_train[:1].astype(numpy.float32), target_opset=12)
    with open(f"{model_name}.onnx", "wb") as f:
        f.write(model_onx.SerializeToString())
    numpy.save(f'{model_name}_predict.npy',y_pred_train)
    print('')
