from flask import Flask, render_template
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
from flask import Flask, jsonify, request,render_template
import pandas as pd
import numpy as np

app = Flask(__name__)


@app.route("/")
def index():
    df = pd.read_json('static/data/Expression.json')
    dfexp = list(df.drop_duplicates(inplace=False, subset="Atlas_Organism_Part", keep='first')['Atlas_Organism_Part'].sort_values(
        ascending=True))

    return render_template("index.html", exp=dfexp)


# Load the Olivetti faces dataset
data = np.load('model/olivetti_faces.npy')
labels = np.load('model/olivetti_faces_target.npy')

# Preprocess the data
data = data.astype('float32') / 255
labels = to_categorical(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2)

# Build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(64, 64)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(40, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10)





@app.route('/faces/<int:target>', methods=['GET'])
def faces(target):
    # Predict the labels of the test data using the trained model
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)

    # Filter the data to get the images that match the selected target label
    indices = np.where(predicted_labels == target)[0][:10]
    images = X_test[indices]

    # Render the images in the HTML template
    return render_template('faces.html', images=images)


@app.route("/genes/<id>/")
def Genes(id):

    df1 = pd.read_json('static/data/Genes.json')
    df2 = pd.read_json('static/data/Transcripts.json')
    df3 = pd.read_json('static/data/ParalogsHuman.json')
    df_genes = df1[df1['Ensembl_Gene_ID'] == id]
    df_transcripts = df2[df2['Ensembl_Gene_ID'] == id]
    df_paralogs = df3[df3['Ensembl_Gene_ID'] == id]

    return render_template("gene.html", lignes1=[df_genes.to_html(classes="table table-stripped", header=True)], lignes2=[df_transcripts.to_html(classes="table table-stripped", header=True)], lignes3=[df_paralogs.to_html(classes="table table-stripped", header=True)], gid=id)


@app.route("/parts/<part>/genes/")
def genespart(part):
    df1 = pd.read_json('static/data/Expression.json')
    df2 = pd.read_json('static/data/Transcripts.json')
    df_exp = df1[df1['Atlas_Organism_Part'] == part]
    df_transcripts = df2.loc[df2['Ensembl_Transcript_ID'].isin(
        df_exp['Ensembl_Transcript_ID'])]
    trans = list(df_transcripts.drop_duplicates(
        subset="Ensembl_Gene_ID", keep='first', inplace=False)['Ensembl_Gene_ID'])

    return render_template("part.html", transcripts=trans, parts=part)


@app.route("/immobilier")
def immobilier():

    df = pd.read_csv('static/data/my_file.csv')
    return render_template("immobilier.html", lignes=[df.to_html(classes="table table-stripped", header=True)])


@app.route("/contact")
def contact():
    return render_template('contact.html')
