import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict
hbp_walks = pd.read_csv('walk_hbp.csv')
hbp_walks = hbp_walks.drop("iso_value", axis=1)
hbp_walks = hbp_walks.drop("woba_value", axis=1)

strikeouts = pd.read_csv('strikeouts.csv')
strikeouts = strikeouts.drop("iso_value", axis=1)
strikeouts = strikeouts.drop("woba_value", axis=1)

model_data =  pd.read_csv("model_stats.csv")
model_data = model_data.drop("iso_value", axis=1)
model_data = model_data.drop("woba_value", axis=1)
needed = ["launch_speed","launch_angle","total_bases"]
model_data = model_data.dropna(subset=needed)
model_data["total_bases"] = model_data['total_bases'].round(1)
mapping = {
    -0.6:0,
    0.5:1,
    1.5:2,
    3.5:4,
    2.5:3
}
model_data["total_bases"] = model_data["total_bases"].replace(mapping)
model_data["total_bases"] = model_data["total_bases"].apply(int)

test_data = pd.read_csv('test_stats.csv')
test_data = test_data.drop("iso_value", axis=1)
test_data = test_data.drop("woba_value", axis=1)
needed = ["launch_speed","launch_angle","total_bases"]
test_data = test_data.dropna(subset=needed)
test_data["total_bases"] = test_data['total_bases'].round(1)
test_data["total_bases"] = test_data["total_bases"].replace(mapping)
test_data["total_bases"] = test_data["total_bases"].apply(int)

# model_data.plot(kind='hexbin', x="launch_speed", y="launch_angle",
#                 C="total_bases", colormap=plt.get_cmap('jet'), colorbar=True)

# plt.show()

x_categories = ['launch_speed', 'launch_angle']
x_train = model_data[x_categories]
y_train = model_data['total_bases']
x_test = test_data[x_categories]
y_test = test_data['total_bases']

################### Model Selection #########################

# percent_outs = y_train.value_counts()[0] / len(y_train)
# param_grid = {
#     'n_neighbors': range(1, 31),
#     'weights': ['uniform', 'distance'],
#     'metric': ['euclidean', 'manhattan']
# }
# grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy')
# grid_search.fit(x_train, y_train)
# print(f"Best Parameters: {grid_search.best_params_}")
# print(f"Best Score: {grid_search.best_score_}")

# knn = KNeighborsClassifier(
#     n_neighbors=20, 
#     weights='uniform')
# knn.fit(x_train, y_train)
# y_train_pred = cross_val_predict(knn, x_train, y_train, cv=5)
# acc_score = accuracy_score(y_train, y_train_pred)
# print(f"Accuracy: {acc_score}")
# print(f"Percent Outs: {percent_outs}")

# conf_mx = confusion_matrix(y_train, y_train_pred)
# plt.matshow(conf_mx, cmap=plt.cm.gray)
# plt.show()
# row_sums = conf_mx.sum(axis=1, keepdims=True)
# norm_conf_mx = conf_mx / row_sums
# np.fill_diagonal(norm_conf_mx, 0)
# plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
# plt.show()

#matrix clearly doesn't do a great job of classifying other things correctly because of the 
#high prominence of outs in the dataset. Instead we'll output probabilities

knn = KNeighborsClassifier(
     n_neighbors=15, 
     weights='uniform')

knn.fit(x_train, y_train)
pred = knn.predict(x_test)
acc_score = accuracy_score(pred, y_test)
print(f"Model Accuracy = {acc_score}")


######### fix this to be more accurate by year#########
all_years_data = pd.concat([x_train, x_test])
prob_predictions = knn.predict_proba(all_years_data)
predicted_wobas = []
for pred in prob_predictions:
    predicted_woba = (
        .883 * pred[1] + 
        1.253 * pred[2] +
        1.587 * pred[3] +
        2.042 * pred[4]
    )
    predicted_wobas.append(predicted_woba)

mlb_data = pd.concat([model_data, test_data])
mlb_data['liam_xwoba'] = predicted_wobas

def walk_woba_from_event(event):
    if event =='walk':
        return .692
    elif event == 'hit_by_pitch':
        return .723
    
hbp_walks['liam_xwoba'] = hbp_walks.apply(
    lambda row: walk_woba_from_event(row['events']),
    axis=1
)

strikeouts['liam_xwoba'] = 0

all_events = pd.concat([mlb_data, hbp_walks, strikeouts])
all_events.to_csv('full_stats.csv', index = False)