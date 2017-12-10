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
    DB = PostgresqlDatabase(os.environ['DATABASE_URL'])
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

