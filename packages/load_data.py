import pandas as pd

def load_countries(): 
   # df = pd.read_csv('./model/data-test.csv')
    df = data = pd.read_excel('./model/data.xlsx')
    data = df[['Pays', 'Annee','RATING']]
    #data['RATING'] = data['RATING'].fillna('No Notation')
    new_arr = []
    for i in range(data.shape[0] -1):
        lis = {}
        lis['Pays'] = data['Pays'][i]
        lis['Annee'] = data['Annee'][i]
        lis['RATING'] = data['RATING'][i]
        new_arr.append(lis)
    return new_arr