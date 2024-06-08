import pandas as pd

from keras.models import Sequential
from keras.layers import Dense

base = pd.read_csv("autos.csv",encoding = 'ISO-8859-1') 

# PRE-PROCESSAMENTO

#verificando repeitcao de nomes do veiculo
base['name'].value_counts( )
base = base.drop('name', axis = 1)
#dropando colunas irrelevantes
base = base.drop('dateCrawled', axis = 1)
base = base.drop('dateCreated', axis = 1)
base = base.drop('nrOfPictures', axis = 1)
base = base.drop('postalCode', axis = 1)
base = base.drop('lastSeen', axis = 1)


#verificando a dispersao de tipos de venda
base['seller'].value_counts()
base = base.drop('seller',axis = 1)

#verificando a dispersao de tipos de oferta
base['offerType'].value_counts()
base = base.drop('offerType',axis = 1)

#i1 representa a primeira inconsistencia(Precos menores que 10)
i1 = base.loc[base.price <=  10]
base = base[base.price > 10]
#i2
i2 = base.loc[base.price > 350000]
base = base[base.price < 350000]

#Tratamento de NaNs
base.loc[pd.isnull(base['vehicleType'])]
base['vehicleType'].value_counts() #mais ultilizado = limousine
base.loc[pd.isnull(base['gearbox'])]
base['gearbox'].value_counts() #mais ultilizado = manuell
base.loc[pd.isnull(base['model'])]
base['model'].value_counts() #mais ultilizado = golf
base.loc[pd.isnull(base['fuelType'])]
base['fuelType'].value_counts() #mais ultilizado = benzin
base.loc[pd.isnull(base['notRepairedDamage'])]
base['notRepairedDamage'].value_counts() # nein


#substituir NaN pelos mais ultilizador
valores = {'vehicleType': 'limousine', 
           'gearbox': 'manuell',
           'model': 'golf', 
           'fuelType': 'benzin',
           'notRepairedDamage': 'nein'}
base = base.fillna(value = valores)

#verifica a quantidade de NaN
base.isnull().sum()

previsores = base.iloc[:,1:13].values
preco_real = base.iloc[:,0].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
labelencoder_previsores = LabelEncoder()

#Transformando classificacoes de strings para numeros
previsores[:,0] = labelencoder_previsores.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder_previsores.fit_transform(previsores[:,1])
previsores[:,3] = labelencoder_previsores.fit_transform(previsores[:,3])
previsores[:,5] = labelencoder_previsores.fit_transform(previsores[:,5])
previsores[:,8] = labelencoder_previsores.fit_transform(previsores[:,8])
previsores[:,9] = labelencoder_previsores.fit_transform(previsores[:,9])
previsores[:,10] = labelencoder_previsores.fit_transform(previsores[:,10])

#transformar as categorias em one hot encode
onehotencoder = OneHotEncoder(categorical_features= [0,1,3,5,8,9,10 ])
previsores = onehotencoder.fit_transform(previsores).toarray()


#REDE E TREINO 

#criando a rede
regressor = Sequential()
#units = (316+1)/2
regressor.add(Dense(units=158, activation = 'relu', input_dim = 316))
regressor.add(Dense(units=158, activation = 'relu'))
regressor.add(Dense(units=1, activation = 'linear'))

regressor.compile(loss = 'mean_absolute_error', optimizer = 'adam', metrics =['mean_absolute_error'])

regressor.fit(previsores, preco_real, batch_size = 300, epochs = 100)

#fazer previsoes
previsores = regressor.predict(previsores)