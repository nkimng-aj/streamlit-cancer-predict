import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle as pickle


def create_model(data):
  X = data.drop(['diagnosis'], axis = 1)
  y = data['diagnosis']

  # scale the data (normalize the data)
  scaler = StandardScaler()
  scaler.fit(data.drop('diagnosis', axis=1))
  scaled_features = scaler.transform(data.drop('diagnosis', axis=1))
  data_feat = pd.DataFrame(scaled_features, columns=data.columns[:-1])

  X = data_feat
  y = data['diagnosis']

  # split the data
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.30, random_state=42)

  # train
  model = LogisticRegression()
  model.fit(X_train, y_train)

  # test model
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print(f"Accuracy of our model: {accuracy: .2f}")
  print(f"Classification report: {classification_report(y_test, y_pred)}\n")

  return model, scaler


def test_model(model):
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print(f"Accuracy: {accuracy: .2f}")
  print(classification_report(y_test, y_pred))


def get_clean_data():
  data = pd.read_csv("data/data.csv") # import dataset
  data.drop(["Unnamed: 32","id"], axis = 1, inplace = True ) # drop ID number column & Unnamed 32 column
  data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0}) # encode the diagnosis variable

  return data


def main():
  data = get_clean_data()
  model, scaler = create_model(data)
  # test_model(model)
  with open('model.pkl','wb') as f:
    pickle.dump(model,f)
  with open('scaler.pkl','wb') as f:
    pickle.dump(scaler,f)


if __name__ == '__main__':
   main()
   