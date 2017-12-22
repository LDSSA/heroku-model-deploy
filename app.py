import os
from flask import Flask, jsonify
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, BooleanField,
)
from playhouse.shortcuts import model_to_dict


########################################
# Begin database stuff

if 'DATABASE_URL' in os.environ:
    db_url = os.environ['DATABASE_URL']
    dbname = db_url.split('@')[1].split('/')[1]
    user = db_url.split('@')[0].split(':')[1].lstrip('//')
    password = db_url.split('@')[0].split(':')[2]
    host = db_url.split('@')[1].split('/')[0].split(':')[0]
    port = db_url.split('@')[1].split('/')[0].split(':')[1]
    DB = PostgresqlDatabase(
        dbname,
        user=user,
        password=password,
        host=host,
        port=port,
    )
else:
    DB = SqliteDatabase('predictions.db')


class Prediction(Model):
    observation_id = IntegerField(unique=True)
    proba = FloatField()
    predicted_class = BooleanField()
    true_class = BooleanField

    class Meta:
        database = DB


if not Prediction.table_exists():
    DB.create_table(Prediction)

# End database stuff
########################################


########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    proba = 0.5
    p = Prediction(observation_id=0, proba=proba, predicted_class=True)
    p.save()
    return str(proba)


@app.route('/update', methods=['POST'])
def update():
    p = Prediction.get(Prediction.observation_id == 0)
    p.true_class = False
    p.save()
    return 'success'


@app.route('/list-db-contents')
def list_db_contents():
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ])

# End webserver stuff
########################################

if __name__ == "__main__":
    app.run()

