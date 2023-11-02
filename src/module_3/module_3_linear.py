# %% Import packages
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt

# %% Load data
address = "D:\\Yo\\"
filename = "feature_frame.csv"
df = pd.read_csv(address + filename)

# %% Data preprocessing

# Obtain the id's for the orders where 5 or more products where bought
filter_df = df[df["outcome"]==1]
filter_df = filter_df.groupby("order_id").filter(lambda x: len(x) > 4)

# Filter the data so we retain only those ids
dataset = df[df["order_id"].isin(filter_df["order_id"])]

# Change date format to retain only year, month and day
dataset["created_at"] = dataset["created_at"].str[:-9]
dataset["created_at"] = dataset["created_at"].str.replace("-","")
pd.to_numeric(dataset["created_at"])
dataset["order_date"] = dataset["order_date"].str[:-9]
dataset["order_date"] = dataset["order_date"].str.replace("-","")
pd.to_numeric(dataset["order_date"])

# Order by date
dataset = dataset.sort_values(by=["order_date"])

# %% Drop qualitative features
print(dataset.columns)
print(dataset.dtypes)
dataset.drop(["product_type" , "vendor"], axis=1, inplace=True)

# %% Divide the data in training, validation and test set
def timesplit_by_order_id(data: pd.DataFrame, train_size: float, validation_test_size: float):
    total_orders = data.order_id.unique()
    train_num = round(total_orders.size*train_size)
    val_num = round(total_orders.size*validation_test_size)
    test_num = val_num
    gap = round(total_orders.size*(1-validation_test_size-train_size)/3)

    train_orders=total_orders[0:train_num]
    val_orders=total_orders[train_num+gap:train_num+gap+val_num]
    test_orders=total_orders[train_num+2*gap+val_num:train_num+2*gap+val_num+test_num]

    train = dataset[dataset["order_id"].isin(train_orders)]
    val = dataset[dataset["order_id"].isin(val_orders)]
    test = dataset[dataset["order_id"].isin(test_orders)]

    return train, val, test

train_set, val_set, test_set = timesplit_by_order_id(dataset, 0.74, 0.10)

X_train = train_set.drop(["outcome"], axis=1)
y_train = train_set["outcome"]
X_val = val_set.drop(["outcome"], axis=1)
y_val = val_set["outcome"]
X_test = test_set.drop(["outcome"], axis=1)
y_test = test_set["outcome"]

# %% Train linear model(I will try a couple of them)

reg1 = linear_model.LinearRegression()
reg1.fit(X_train, y_train)

reg2 =linear_model.Ridge()
reg2.fit(X_train, y_train)

# %% Train non-linear models (tree based)

tree1 = RandomForestClassifier(max_depth=2, random_state=0)
tree1.fit(X_train, y_train)

tree2 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.3,
max_depth=2, random_state=0)
tree2.fit(X_train, y_train)

# %% Calculate metrics
# First create an array with the baseline and predictions
y=[]
y.append(test_set["global_popularity"])
y.append(reg1.predict(X_test))
y.append(reg2.predict(X_test))
y.append(tree1.predict(X_test))
y.append(tree2.predict(X_test))

#calculate precision, recall, FPR and TPR
precision = [None]*len(y)
recall = [None]*len(y)
FPR = [None]*len(y)
TPR = [None]*len(y)
for i, y in enumerate(y):
    precision[i], recall[i], _ = metrics.precision_recall_curve(y_test, y)
    FPR[i], TPR[i], _ = metrics.roc_curve(y_test, y)

# %% Plot metrics
model_name = ["Baseline", 
              "LSR Regression", 
              "Ridge Regression", 
              "Random Forest", 
              "Boosted Trees"]
#plot precision-recall and ROC curve
fig, axs = plt.subplots(1, 2, figsize = (12, 6))
for i, prec in enumerate(precision):
    axs[0].plot(recall[i], prec)
    axs[1].plot(FPR[i], TPR[i], label=model_name[i])

axs[0].set_title('Precision-Recall Curve')
axs[0].set_ylabel('Precision')
axs[0].set_xlabel('Recall')

axs[1].set_title('ROC curve')
axs[1].set_ylabel('True Positive Rate')
axs[1].set_xlabel('False Positive Rate')

fig.legend()
plt.savefig("metrics.png")

# %%
