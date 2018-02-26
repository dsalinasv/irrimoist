import pandas as pd
import json
import time
from sklearn.externals import joblib
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

model_file_name = 'model.pkl'
model_columns_file_name = 'model_columns.pkl'

clf = joblib.load(model_file_name)
model_columns = joblib.load(model_columns_file_name)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if clf:
        data = request.data
        query = pd.get_dummies(pd.DataFrame(json.loads(data)))
        query = query.reindex(columns=model_columns, fill_value=0)
        prediction = list(clf.predict(query))
        return jsonify({'prediction': prediction})
    return 'fails'

@app.route('/train', methods=['GET'])
def train():
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    training_data = 'sensor.csv'
    include = ['Sensor', 'Textura', 'Periodo', 'Humedad']
    dependent_variable = include[-1]
    df = pd.read_csv(training_data)
    df_ = df[include]
    categoricals = []
    for col, col_type in df_.dtypes.iteritems():
        if col_type == 'O':
            categoricals.append(col)
        else:
            df_[col].fillna(0, inplace=True)
    df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)
    X = df_ohe[df_ohe.columns.difference([dependent_variable])]
    y = df_ohe[dependent_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    model_columns = list(X.columns)
    joblib.dump(model_columns, model_columns_file_name)
    clf = LinearRegression()
    start = time.time()
    clf.fit(X_train,y_train)
    success = jsonify([{'Time' : (time.time() - start)} , {'Score' : clf.score(X, y)}])
    joblib.dump(clf, model_file_name)
    return success

if __name__ == '__main__':
    app.run()