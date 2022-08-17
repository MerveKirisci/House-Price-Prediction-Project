#####################################################
# HOUSE PRICE PREDICTION
#####################################################


# SalePrice - mülkün dolar cinsinden satış fiyatı. Bu, tahmin etmeye çalışılan hedef değişkendir.
# MSSubClass: İnşaat sınıfı
# MSZoning: Genel imar sınıflandırması
# LotFrontage: Mülkiyetin cadde ile doğrudan bağlantısının olup olmaması
# LotArea: Parsel büyüklüğü
# Street: Yol erişiminin tipi
# Alley: Sokak girişi tipi
# LotShape: Mülkün genel şekli
# LandContour: Mülkün düzlüğü
# Utulities: Mevcut hizmetlerin türü
# LotConfig: Parsel yapılandırması
# LandSlope: Mülkün eğimi
# Neighborhood: Ames şehir sınırları içindeki fiziksel konumu
# Condition1: Ana yol veya tren yoluna yakınlık
# Condition2: Ana yola veya demiryoluna yakınlık (eğer ikinci bir yer varsa)
# BldgType: Konut tipi
# HouseStyle: Konut sitili
# OverallQual: Genel malzeme ve bitiş kalitesi
# OverallCond: Genel durum değerlendirmesi
# YearBuilt: Orijinal yapım tarihi
# YearRemodAdd: Yeniden düzenleme tarihi
# RoofStyle: Çatı tipi
# RoofMatl: Çatı malzemesi
# Exterior1st: Evdeki dış kaplama
# Exterior2nd: Evdeki dış kaplama (birden fazla malzeme varsa)
# MasVnrType: Duvar kaplama türü
# MasVnrArea: Kare ayaklı duvar kaplama alanı
# ExterQual: Dış malzeme kalitesi
# ExterCond: Malzemenin dışta mevcut durumu
# Foundation: Vakıf tipi
# BsmtQual: Bodrumun yüksekliği
# BsmtCond: Bodrum katının genel durumu
# BsmtExposure: Yürüyüş veya bahçe katı bodrum duvarları
# BsmtFinType1: Bodrum bitmiş alanının kalitesi
# BsmtFinSF1: Tip 1 bitmiş alanın metre karesi
# BsmtFinType2: İkinci bitmiş alanın kalitesi (varsa)
# BsmtFinSF2: Tip 2 bitmiş alanın metre karesi
# BsmtUnfSF: Bodrumun bitmemiş alanın metre karesi
# TotalBsmtSF: Bodrum alanının toplam metre karesi
# Heating: Isıtma tipi
# HeatingQC: Isıtma kalitesi ve durumu
# CentralAir: Merkezi klima
# Electrical: elektrik sistemi
# 1stFlrSF: Birinci Kat metre kare alanı
# 2ndFlrSF: İkinci kat metre kare alanı
# LowQualFinSF: Düşük kaliteli bitmiş alanlar (tüm katlar)
# GrLivArea: Üstü (zemin) oturma alanı metre karesi
# BsmtFullBath: Bodrum katındaki tam banyolar
# BsmtHalfBath: Bodrum katındaki yarım banyolar
# FullBath: Üst katlardaki tam banyolar
# HalfBath: Üst katlardaki yarım banyolar
# BedroomAbvGr: Bodrum seviyesinin üstünde yatak odası sayısı
# KitchenAbvGr: Bodrum seviyesinin üstünde mutfak Sayısı
# KitchenQual: Mutfak kalitesi
# TotRmsAbvGrd: Üst katlardaki toplam oda (banyo içermez)
# Functional: Ev işlevselliği değerlendirmesi
# Fireplaces: Şömineler
# FireplaceQu: Şömine kalitesi
# Garage Türü: Garaj yeri
# GarageYrBlt: Garajın yapım yılı
# GarageFinish: Garajın iç yüzeyi
# GarageCars: Araç kapasitesi
# GarageArea: Garajın alanı
# GarageQual: Garaj kalitesi
# GarageCond: Garaj durumu
# PavedDrive: Garajla yol arasındaki yol
# WoodDeckSF: Ayaklı ahşap güverte alanı
# OpenPorchSF: Kapı önündeki açık veranda alanı
# EnclosedPorch: Kapı önündeki kapalı veranda alan
# 3SsPorch: Üç mevsim veranda alanı
# ScreenPorch: Veranda örtü alanı
# PoolArea: Havuzun metre kare alanı
# PoolQC: Havuz kalitesi
# Fence: Çit kalitesi
# MiscFeature: Diğer kategorilerde bulunmayan özellikler
# MiscVal: Çeşitli özelliklerin değeri
# MoSold: Satıldığı ay
# YrSold: Satıldığı yıl
# SaleType: Satış Türü
# SaleCondition: Satış Durumu


# Kütüphanelerin import edilmesi

import numpy as np
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score, validation_curve
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Görünüm ayarları
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Veri setinin okutulması
train_df = pd.read_csv("datasets/house_price_prediction_train.csv")
test_df = pd.read_csv("datasets/house_price_prediction_test.csv")

# Veri setlerinin birleştirilmesi
df = train_df.append(test_df).reset_index(drop=True)


##############################
# EDA
##############################

# 1. Genel Resim
# 2. Kategorik Değişken Analizi
# 3. Sayısal Değişken Analizi
# 4. Hedef Değişken Analizi
# 5. Korelasyon Analizi


################
# 1. Genel Resim
################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head(head))

    print("##################### Tail #####################")
    print(dataframe.tail(head))

    print("##################### is null? #####################")
    print(dataframe.isnull().sum())

    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print(dataframe.describe().T)


check_df(df)

df = df.loc[df["SalePrice"]<=400000]


# Kategorik ve Numeric değişkenlerin belirlenmesi
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.
    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri
    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi
    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))
    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


################
# 2. Kategorik Değişken Analizi
################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("###########################################")
    if plot:
        sns.countplot(x = dataframe[col_name], data= dataframe)
        plt.show()

 for col in cat_cols:
        cat_summary(df, col)


################
# 3. Sayısal Değişken Analizi
################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, True)

################
# 4. Hedef Değişken Analizi
################

# Kategorik değişkenler için

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}),
          end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df,"SalePrice",col)


# Numeric değişkenler için ;
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:"mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df,"SalePrice",col)


################
# 5. Korelasyon Analizi
################

corr_matrix = df.corr()
threshold = 0.5
filtre = np.abs(corr_matrix["SalePrice"]) > threshold
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(df[corr_features].corr(), annot=True, fmt=".2f", cmap="inferno")
plt.title("Correlation of Features and Corr Threshold")
plt.show()

#Yüksek korelasyona sahip değişkenler;
def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdPu")
        plt.show()
    return drop_list
high_correlated_cols(df,plot=True)

######################################
# Veri ön işleme
######################################

# 1. Aykırı Değer Analizi
# 2. Eksik Değer Analizi
# 3. Feature Engineering (Özellik Mühendisliği)
# 4. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
# 5. Feature Scaling (Özellik Ölçeklendirme)

################
# 1. Aykırı Değer Analizi
################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# Outlier olup olmadığını kontrol ediyoruz
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# Aykırı Değer kontrol ediyoruz
for col in num_cols:
    if col != "SalePrice":
      print(col, check_outlier(df, col))

# Var olan aykırı değerleri threshold değerleri ile yer değiştiriyoruz
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Aykırı değerlerin baskılanması
for col in num_cols:
    if col != "SalePrice":
        replace_with_thresholds(df,col)

################
# 2. Eksik Değer Analizi
################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns
missing_values_table(df)

# ratio: kolon içerisinde yüzde kaç eksik gözlem olduğunu verir
# n_miss: her bir kolondaki eksik gözlem sayısını verir

# BsmtQual: Bodrumun yüksekliği
# BsmtCond: Bodrum katının genel durumu
# BsmtExposure: Yürüyüş veya bahçe katı bodrum duvarları
# BsmtFinType1: Bodrum bitmiş alanının kalitesi
# BsmtFinType2: İkinci bitmiş alanın kalitesi (varsa)
# Bazı değişkenlerdeki boş değerler evin o özelliğe sahip olmadığını ifade etmektedir

no_cols = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","FireplaceQu",
           "GarageType","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature"]

# Kolonlardaki boşlukların "No" ifadesi ile doldurulması
for col in no_cols:
    df[col].fillna("No",inplace=True)

missing_values_table(df)

# Eksik değerleri mean veya median değerleri ile doldurma işlemi
def quick_missing_imp(data, num_method="median", cat_length=20, target="SalePrice"):
    variables_with_na = [col for col in data.columns if data[col].isnull().sum() > 0]  # Eksik değere sahip olan değişkenler listelenir

    temp_target = data[target]

    print("# BEFORE")
    print(data[variables_with_na].isnull().sum(), "\n\n")  # Uygulama öncesi değişkenlerin eksik değerlerinin sayısı

    # değişken object ve sınıf sayısı cat_lengthe eşit veya altındaysa boş değerleri mode ile doldur
    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x, axis=0)

    # num_method mean ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    # num_method median ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data


df = quick_missing_imp(df, num_method="median", cat_length=17)
df.columns

df["SalePrice"].mean()
# 174873.43854748603

df["SalePrice"].std()
# 65922.70393689284

#######
# Rare
#######

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
rare_analyser(df, "SalePrice", cat_cols)


def rare_encoder(dataframe, rare_perc, cat_cols):
    rare_columns = [col for col in cat_cols if (dataframe[col].value_counts() / len(dataframe) < 0.01).sum() > 1]

    for col in rare_columns:
        tmp = dataframe[col].value_counts() / len(dataframe)
        rare_labels = tmp[tmp < rare_perc].index
        dataframe[col] = np.where(dataframe[col].isin(rare_labels), "Rare", dataframe[col])
    return dataframe

# Nadir sınıfların tespit edilmesi
df = rare_encoder(df, 0.01, cat_cols)

rare_analyser(df, "SalePrice", cat_cols)


useless_cols = [col for col in cat_cols if df[col].nunique() == 1 or
                (df[col].nunique() == 2 and (df[col].value_counts() / len(df) <= 0.01).any(axis=None))]
# ['Street', 'Utilities', 'PoolQC', 'PoolArea'] -> useless kolona eklenecek


useless_cols = ['Street', 'Utilities', 'PoolQC', 'PoolArea',"MiscFeature","Alley","LandContour","LandSlope", 'Neighborhood']

# useless_cols'teki değişkenlerin düşürülmesi
df.drop(useless_cols, axis=1, inplace=True)

useless_cols



################
# 3. Feature Engineering (Özellik Mühendisliği)
################

### renovated building age ###
# YearBuilt: Original construction date
# YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)
df["new_remodadd_built"] = df["YearRemodAdd"] - df["YearBuilt"]

### Bath Numbers ###
# BsmtFullBath: Basement full bathrooms
# BsmtHalfBath: Basement half bathrooms
# FullBath: Full bathrooms above grade
# HalfBath: Half baths above grade
df["total_bath"] = df["BsmtHalfBath"] + df["BsmtFullBath"]  + df["HalfBath"]

### Total Floor ###
# 1stFlrSF: First Floor square feet
# 2ndFlrSF: Second floor square feet
# TotalBsmtSF: Total square feet of basement area
# GrLivArea: Above grade (ground) living area square feet
df["new_total_floor"] = df["1stFlrSF"] + df["2ndFlrSF"]
df["new_living_area"] = df["TotalBsmtSF"] + df["GrLivArea"]

### Garage ###
# GarageQual: Garage quality
# GarageCond: Garage condition
df["GarageQual"].value_counts()
df["GarageCond"].value_counts()
df["TotalGarageQual"] = df[["GarageQual", "GarageCond"]].sum(axis = 1)
df["TotalGarageQual"]


### House Age ###
# YrSold: Year Sold (YYYY)
# YearBuilt: Original construction date
# YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)

df["house_age"] = df["YrSold"] - df["YearBuilt"]
df["restoration_age"] = df["YrSold"] - df["YearRemodAdd"]
df["garage_age"] = df["GarageYrBlt"] - df["YearBuilt"]
df["garage_sold"] = df["YrSold"] - df["GarageYrBlt"]

### Living area and lot size ###
# GrLivArea: Above grade (ground) living area square feet
# LotArea: Lot size in square feet
# TotalBsmtSF: Total square feet of basement area
df["new_GrLivArea_LotArea"] = df["GrLivArea"] / df["LotArea"]
df["total_living_area"] = df["TotalBsmtSF"] + df["GrLivArea"]

### Lot Ratio ###
# GrLivArea: Above grade (ground) living area square feet
# LotArea: Lot size in square feet
df["LotRatio"] = df["GrLivArea"] / df["LotArea"]
df["RatioArea"] = (df["TotalBsmtSF"] + df["new_total_floor"]) / df["LotArea"]



################
# 4. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
################

# One-Hot Encoding
cat_cols, cat_but_car, num_cols = grab_col_names(df)

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and len(df[col].unique()) == 2]

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_cols:
    label_encoder(df, col)


df = one_hot_encoder(df, cat_cols, drop_first=True)

##################################
# Modeling
##################################

y = df['SalePrice']
X = df.drop(["Id", "SalePrice"], axis=1)

# Veriyi test ve train seti olarak ikiye bölelim

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

#BASE MODEL

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor())]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")


# RMSE: 25032.3298 (LR)
# RMSE: 23864.6005 (Ridge)
# RMSE: 24863.2636 (Lasso)
# RMSE: 26874.1209 (ElasticNet)
# RMSE: 36149.7212 (KNN)
# RMSE: 37034.613 (CART)
# RMSE: 24911.5067 (RF)
# RMSE: 23468.5436 (GBM)
# RMSE: 24750.7129 (XGBoost)
# RMSE: 23192.6355 (LightGBM)


# Hyperparameter Optimization
lgbm_model = LGBMRegressor(random_state=46)

rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model, X, y, cv=5, scoring="neg_mean_squared_error")))

lgbm_params = {"learning_rate": [0.01, 0.1, 0.05],
               "n_estimators": [1500, 3000, 6000],
               "colsample_bytree": [0.5, 0.7],
               "num_leaves": [31, 35],
               "max_depth": [3, 5]}

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X_train, y_train)

final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

y_pred = final_model.predict(X_test)

rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))
# 21993.624868240062

lgbm_model = LGBMRegressor(learning_rate = 0.01, random_state=46)
