# CafsPy
A covering Array Feature Selection Library. Aditioanlly, in order to use this librabry is recomended to intstall it over a new python virtual enviroment without any **pre-intsalled** library  to avoid conflicts. 

Instruction to install: 

First, install from pip. 
```
pip install cafspy
```
Once installed, look at the follow mini tutorial, in order to use CafsPy accordingly. There are 2 Covering Array algorithms available.

```
from cafspy import ICAFS
from cafspy import CAFS
```

Then, in order to use ICAFS, we need to set up the datasets that you want to generate features subsets. For this sample , we will use Non-plus/ plus algarrobo available in this project. Is important to remark that the *X* dataset must containts string header and numerics values. The same must be taken into account for **y**. 

```
df_algarrobo = pd.read_csv('data/algarrobo.csv') 
unique_names_algarrobo = df_algarrobo['Labels'].unique()

algarrobo_x = df_algarrobo.loc[:, 'R':'REDVI']

y_algarrobo = df_algarrobo['Labels'].replace(to_replace=unique_names_algarrobo, value=[0, 1]).to_frame()

X_algarrobo = (algarrobo_x-algarrobo_x.min())/(algarrobo_x.max()-algarrobo_x.min())

```
Once the data is prepared, is important to define the classfication algorithm from sklearn that you want to work with. For this little sample, we will wotk with *kNN* .

```
from sklearn.neighbors import KNeighborsClassifier
lr_algo = KNeighborsClassifier(n_neighbors=3)
```

Secondly, we need to definte the interaction **t**. Remember that this represent the interaction between any interaction, this means that it will generate all subsets of size **t** btween any feature available. Hence, the complexity will increase. for that reason on the following papaer a deep anaylisis was done to get the best interaction level. This sample will work on **t=2**, and number of iteration **T=10**. The **print_logs** will show the best features selected at each iteration. 

```
  scores_list,feature_list = ICAFS(X_algarrobo,y_algarrobo,t=2,T=10,lr=lr_algo,print_logs=True)
```

for this sample, this shows the best feature selected during **10** iterations

```
f1 score= 0.7351370100315423, iteration:1, numbers features selected =9,best features selected=NGRDI, NDVI, RVI, DVI, EVI, REVI, NDRE, RERVI, REDVI

best f1 score= 0.7216630322908072, iteration:2, numbers features selected =6,best features selected=NGRDI, NDVI, RVI, EVI, NDRE, REDVI

best f1 score= 0.717571263976013, iteration:3, numbers features selected =4,best features selected=NGRDI, NDVI, RVI, NDRE

best f1 score= 0.7092041147580356, iteration:4, numbers features selected =3,best features selected=NGRDI, NDVI, RVI

best f1 score= 0.7092041147580356, iteration:5, numbers features selected =3,best features selected=NGRDI, NDVI, RVI

best f1 score= 0.7092041147580356, iteration:6, numbers features selected =3,best features selected=NGRDI, NDVI, RVI

best f1 score= 0.7092041147580356, iteration:7, numbers features selected =3,best features selected=NGRDI, NDVI, RVI

best f1 score= 0.7092041147580356, iteration:8, numbers features selected =3,best features selected=NGRDI, NDVI, RVI

best f1 score= 0.7092041147580356, iteration:9, numbers features selected =3,best features selected=NGRDI, NDVI, RVI

best f1 score= 0.7092041147580356, iteration:10, numbers features selected =3,best features selected=NGRDI, NDVI, RVI
```

Another important parameter is the **shuffle** and **seed**, when acitvated and configured, different result will be gotten. 