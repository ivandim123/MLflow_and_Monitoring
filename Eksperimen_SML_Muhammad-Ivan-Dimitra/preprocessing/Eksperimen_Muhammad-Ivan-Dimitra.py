#!/usr/bin/env python
# coding: utf-8

# # **1. Perkenalan Dataset**
# 

# **Sumber Dataset**:  https://www.kaggle.com/datasets/ritwikb3/heart-disease-statlog?resource=download
# 
# ## Konteks
# Dataset yang digunakan adalah Statlog Heart Disease yang diambil dari repositori UCI. Dataset ini berisi data dari 270 individu. Terdapat 14 kolom dalam dataset ini (yang telah diekstrak dari kumpulan data yang lebih besar, yaitu 75 kolom). Tidak ada nilai yang hilang (missing values). Tugas klasifikasi adalah untuk memprediksi apakah seseorang menderita penyakit jantung atau tidak.
# Nilai target:
# 
# 0: tidak menderita penyakit jantung (absence)
# 
# 1: menderita penyakit jantung (presence)
# 
# Data asli:
# https://archive.ics.uci.edu/ml/datasets/statlog+(heart)
# 
# ## Isi Dataset
# Database ini terdiri dari 13 atribut dan 1 variabel target. Terdiri dari 8 nilai nominal dan 5 nilai numerik. Deskripsi rinci setiap fitur sebagai berikut:
# 
# - age: Usia pasien dalam tahun (Numerik)
# 
# - sex: Jenis kelamin (Laki-laki: 1; Perempuan: 0) 
# 
# - cp (chest pain): Jenis nyeri dada yang dialami pasien, terbagi menjadi 4 kategori:
# 
#       0: angina tipikal
# 
#       1: angina atipikal
# 
#       2: nyeri non-angina
# 
#       3: asimptomatik
#       
# 
# - trestbps: Tekanan darah saat istirahat dalam mm/HG (Numerik)
# 
# - chol: Kadar kolesterol serum dalam mg/dl (Numerik)
# 
# - fbs (fasting blood sugar): Kadar gula darah puasa > 120 mg/dl
# 
#       1: benar
# 
#       0: salah
#       
# 
# - restecg: Hasil elektrokardiogram saat istirahat, memiliki 3 nilai:
# 
#       0: Normal
# 
#       1: Kelainan gelombang ST-T (inversi T dan/atau elevasi atau depresi ST > 0.05 mV)
# 
#       2: Hipertrofi ventrikel kiri berdasarkan kriteria Estes
#       
# 
# - thalach: Detak jantung maksimum yang dicapai (Numerik)
# 
# - exang: Angina yang diinduksi oleh olahraga
# 
#       0: Tidak
# 
#       1: Ya
#       
# 
# - oldpeak: Depresi ST yang diinduksi oleh olahraga relatif terhadap kondisi istirahat (Numerik)
# 
# - slope: Kemiringan segmen ST selama puncak olahraga
# 
#       0: naik (upsloping)
# 
#       1: datar (flat)
# 
#       2: turun (downsloping)
#       
# 
# - ca: Jumlah pembuluh darah besar (0–3) yang terlihat melalui fluoroskopi 
# 
# - thal: Gangguan darah yang disebut thalassemia
# 
#       0: NULL
# 
#       1: Aliran darah normal
# 
#       2: Cacat tetap (tidak ada aliran darah di sebagian jantung)
# 
#       3: Cacat reversibel (ada aliran darah tetapi tidak normal)
#       
# 
# - target: Variabel target yang harus diprediksi
# 
#       0: Tidak menderita penyakit jantung
# 
#       1: Menderita penyakit jantung
# 
# Variabel yang Diprediksi
# Apakah seseorang tidak menderita (0) atau menderita (1) penyakit jantung.
# 
# 
# ## Ucapan Terima Kasih
# Beberapa publikasi yang mengutip penggunaan dataset ini:
# 
# - Gavin Brown. Diversity in Neural Network Ensembles. University of Birmingham, 2004.
# 
# - Igor Kononenko, Edvard Simec, Marko Robnik-Sikonja. Overcoming the Myopia of Inductive Learning Algorithms with RELIEFF. Appl. Intell, 7. 1997.
# 
# - Alexander K. Seewald. Dissertation Towards Understanding Stacking Studies of a General Ensemble Learning Scheme.
# 
# - Elena Smirnova, Ida G. Sprinkhuizen-Kuyper, I. Nalbantis, ERIM, Universiteit Rotterdam. Unanimous Voting using Support Vector Machines. IKAT, Universiteit Maastricht.
# 

# # **2. Import Library**

# In[1]:


# Manipulasi dan Analisis Data
import pandas as pd
import numpy as np


# Visualisasi Data
import matplotlib.pyplot as plt
import seaborn as sns


# Pra-pemrosesan & Evaluasi
from sklearn.preprocessing import StandardScaler


# Utilities
import warnings
warnings.filterwarnings('ignore')


# # **3. Memuat Dataset**

# In[2]:


# Memuat dataset Heart_disease_statlog.csv
df = pd.read_csv('../Heart_disease_statlog.csv')
df.head()


# # **4. Exploratory Data Analysis (EDA)**
# 
# Pada tahap ini, Anda akan melakukan **Exploratory Data Analysis (EDA)** untuk memahami karakteristik dataset.
# 
# Tujuan dari EDA adalah untuk memperoleh wawasan awal yang mendalam mengenai data dan menentukan langkah selanjutnya dalam analisis atau pemodelan.

# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


# Cek missing values (NA) dan data duplikat
print("Jumlah missing values tiap kolom:")
print(df.isna().sum())
print("\nJumlah data duplikat:", df.duplicated().sum())


# In[6]:


plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# In[7]:


# Membuat plot distribusi target terhadap setiap fitur
features = [col for col in df.columns if col != 'target']
plt.figure(figsize=(18, 30))
for i, feature in enumerate(features, 1):
    plt.subplot(5, 3, i)
    if df[feature].nunique() <= 10:
        sns.countplot(x=feature, hue='target', data=df)
    else:
        sns.histplot(data=df, x=feature, hue='target', kde=True, element='step', stat='density', common_norm=False)
    plt.title(f'Target vs {feature}')
plt.tight_layout()
plt.show()


# In[8]:


# Menggunakan boxplot untuk mendeteksi outlier pada setiap fitur numerik
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_features.remove('target')

plt.figure(figsize=(18, 10))
for idx, col in enumerate(numerical_features, 1):
    plt.subplot(3, 5, idx)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot {col}')
plt.tight_layout()
plt.show()


# # **5. Data Preprocessing**

# In[9]:


# Capping outlier untuk fitur 'trestbps', 'chol', 'thalach', dan 'oldpeak' menggunakan IQR
capping_features = ['trestbps', 'chol', 'thalach', 'oldpeak']

for feature in capping_features:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_cap = Q1 - 1.5 * IQR
    upper_cap = Q3 + 1.5 * IQR
    df[feature] = np.where(df[feature] < lower_cap, lower_cap,
                           np.where(df[feature] > upper_cap, upper_cap, df[feature]))


# In[10]:


# Mengecek outlier pada fitur yang telah dicapping menggunakan boxplot
plt.figure(figsize=(12, 6))
for idx, feature in enumerate(capping_features, 1):
    plt.subplot(2, 2, idx)
    sns.boxplot(y=df[feature])
    plt.title(f'Boxplot {feature} (Setelah Capping)')
plt.tight_layout()
plt.show()


# In[11]:


# Standarisasi fitur numerik yang sebelumnya diplot menggunakan density (bukan count)
density_features = [col for col in features if df[col].nunique() > 10]

scaler = StandardScaler()
df[density_features] = scaler.fit_transform(df[density_features])

df[density_features].head()


# In[13]:


# Simpan dataframe hasil olahan ke file CSV baru
df.to_csv('HDS_preprocessing.csv', index=False)

