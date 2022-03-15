# Tour Package Predictor
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
data = pd.read_excel('tour_flask.xlsx')
y = pd.DataFrame(data['ProdTaken'])
x = data[['Age','MonthlyIncome', 'DurationOfPitch', 'NumberOfTrips','Passport']]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.20)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(bootstrap = True, max_features = 2, n_estimators = 100, oob_score= False)
#Fitting the model
m=rfc.fit(x_train, y_train)
#Saving the model to disk
pickle.dump(rfc,open('model.pkl','wb') )

