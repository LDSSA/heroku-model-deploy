## IMPORTANT NOTE

This repo is still in development! You may look for now but until I give the signal, it could change at any moment!

# Scikit model behind flask app on heroku

This is a very simplistic yet effective way to deploy a scikit binary
classifier behind a HTTP server on heroku.

There are 4 main topics to cover here

1. Serialization
    - This is covered in notebooks
1. Flask
    - Covered here in the readme
1. Database connection
    - Covered here in the readme
1. Deployment to heroku
    - Also covered here in the readme

### Before continuing

Topic #1 is the only one that is not covered here in this readme. It is covered in two notebooks
that you must read before moving on with the rest of this README.

[Notebook #1](https://github.com/LDSSA/heroku-example/blob/master/Train%20and%20Serialize.ipynb) has
to do with training and serializing a scikit model as well as how to prepare a new observation
that arrives for prediction.

[Notebook #2](https://github.com/LDSSA/heroku-example/blob/master/Deserialize%20and%20use.ipynb) has
to do with deserialization so that you can re-use a model on new observations without having to re-train
it.

## Flask

Have you already read and understood the notebooks on serialiation? Have you already tested your understanding
by pickling and un-pickling your scikit model? Yes yes? Alrighty then, you may continue on.

### What is flask

[Flask](http://flask.pocoo.org/) is an HTTP micro-framework. It is a very minimal code library that allows
for quick and simple HTTP server development and is a great alternative to bigger frameworks like Django.
However, be wary before moving forward with a big project using flask - it can get out of hand very quickly
without the enforced structure that other heavier frameworks like Django provide.

For us, since we only need a total of two endpoints and it doesn't even need to be RESTful, we can stick with
flask and be reasonably justified in it.

### First steps

In order to use flask, you will need to be writing some code in a regular
python file - no more notebooks here. The first step (assuming you have already
done `pip install flask` is to import it at the top of the file. Let's pretend
that we are working in a file called `app.py`

```py
# the request object does exactly what the name suggests: holds
# all of the contents of an HTTP request that someone is making
# the Flask object is for creating an HTTP server - you'll
# see this a few lines down.
# the jsonify function is useful for when we want to return
# json from the function we are using.
from flask import Flask, request, jsonify

# here we use the Flask constructor to create a new
# application that we can add routes to
app = Flask(__name__)
```

This server doesn't do anything yet. In order to make it do stuff we will
need to add http endpoints to it.

### Making HTTP endpoints

With flask, creating an http endpoint is incredibly simple assuming that we already
have the `app` object created from the `Flask` constructor. Let's make a single
endpoint that will serve the predictions:

```py
@app.route('/predict', methods=['POST'])
def predict():
    prediction = 0.5
    return jsonify({
        'prediction': prediction
    })
```

The above route that we have isn't very smart in that it returns the same
prediction every time (0.5) and it doesn't actually care about the input
that you sent it, we've almost created an entire server that serves a prediction!

### Making a complete server

So putting it all together with a few lines of code at the end to start
the server in development mode, we've created an entire server that 
can be run by executing `python app.py`:

```py
# these contents can be put into a file called app.py
# and run by executing:
# python app.py

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    prediction = 0.5
    return jsonify({
        'prediction': prediction
    })
    
if __name__ == "__main__":
    app.run(debug=True)

```

So if you are running this server and you execute the following, you'll get a prediction:

```bash
~ > curl -X POST http://localhost:5000/predict
{
  "prediction": 0.5
}
```

Alright, now that we can run a full flask server, let's try to make something a bit more
useful by receiving new data.

## Receiving a new observation

So now that we've got a way to build an entire server, let's try to actually use the
server to receive new information. There's a pretty nice way to do this via the
[get_json](http://flask.pocoo.org/docs/0.12/api/#flask.Request.get_json) flask function.

For this server, let's say that the model only takes a single field called `unemployed`
and returns `true` if `unemployed` is true and `false` otherwise. The server would now
look like this:

```py
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json()
    at_risk = payload['unemployed']
    return jsonify({
        'prediction': at_risk
    })
    
if __name__ == "__main__":
    app.run(debug=True)
```

You can see the output with the following examples:
```bash
~ > curl -X POST http://localhost:5000/predict -d '{"unemployed": true}' -H "Content-Type:application/json"
{
  "prediction": true
}
```

```bash
~ > curl -X POST http://localhost:5000/predict -d '{"unemployed": false}' -H "Content-Type:application/json"
{
  "prediction": false
}
```

Take a quick note that we had to supply a header of `Content-Type:application/json`
and json data of `{"unemployed": false}`.

## Integrating with a scikit model

Now that we know how to get a python dictionary via the flask `get_json`
function, we're at a point in which we can pick up where the last tutorial
notebook left off! Let's tie it all together by

1. Deserializing the model, columns, and dtypes
1. Turn the new observation into a pandas dataframe
1. Call predict_proba to get liklihood of survival of new observation

### Deserialize model, prep observation, predict

```py
import json
import pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)


with open('columns.json') as fh:
    columns = json.load(fh)

with open('pipeline.pickle', 'rb') as fh:
    pipeline = pickle.load(fh)

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json()
    obs = pd.DataFrame([payload], columns=columns).astype(dtypes)
    proba = pipeline.predict_proba(obs)[0, 1]
    return jsonify({
        'prediction': proba
    })


if __name__ == "__main__":
    app.run(debug=True)
```

Check out how we have now taken the payload and turned it into
a new observation that is a single entry in a dataframe
and can be consumed by the pipeline to be turned into a prediction
of survival. You can see the output with the following:

```
~ >  curl -X POST http://localhost:5000/predict -d '{"Age": 22.0, "Cabin": null, "Embarked": "S", "Fare": 7.25, "Parch": 0, "Pclass": 3, "Sex": "male", "SibSp": 1}' -H "Content-Type:application/json"
{
  "prediction": 0.09264179297127445
}
```

## Keeping track of your predictions

Okay now that you can get data, produce predictions, and return those predictions,
you will need to keep track of what you've been saying about who. Said another way:
you can't just provide predictions and then just forget about it all. You need to
take record of what you have predicted about who so that you can revisit later on
to do some additional analysis on your "through the door" population.

In order to do this, we will need to start working with a database. The database
will keep track of the observations, the predictions we have provided for them
as well as the true outcomes should we be luckly enough to find out.

### ORMs and peewee

When working with databases in code, you generally want to be using a layer of abstraction
called an [ORM](https://en.wikipedia.org/wiki/Object-relational_mapping). For this
exercise we will use a very simplistic ORM called [peewee](http://docs.peewee-orm.com/en/latest/index.html).
This will allow us to use a local database called [sqlite](https://en.wikipedia.org/wiki/SQLite)
when we are developing on our laptops and use a more production-ready database called 
[postgresql](https://en.wikipedia.org/wiki/PostgreSQL) when deploying to heroku with very
little change to our code.

One cool thing that ORMs allow us to do is define the data model that we want
to use in code. So let's use peewee to create a data model to keep track of
predictions and the probabilities we have assigned to them. Once again, we can 
take care of this in a few lines of code:

```py
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, BooleanField, TextField,
)

DB = SqliteDatabase('predictions.db')

class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB

DB.create_tables([Prediction], safe=True)
```

Now we need to take a moment to understand exactly how much these
few lines of code have done for us because it it A LOT.

#### Connect to database

`DB = SqliteDatabase('predictions.db')`

Create a sqlite databse that will be stored in a file called `predictions.db`.
This may seem trivial right now, but soon enough you will see that changing
out this line of code for one other will result in a lot of value for
the effort.

#### Define the data model

`Class Prediction(Model)...`

Define the data model that we will work with. The model has sections for
the following:

- `observation_id`
    - There must be a unique identifier to all observations and it is
      the responsibility of the person providing the observation to give
      this id.
- `observation`
    - We should record the observation itself when it comes in in case
      we want to retrain our model later on.
- `proba`
    - The probability of survival that we assigned
- `true_class`
    - This is for later on in the case where we actually find out what
       actually happened to the observation in which we supplied the
       prediction for.

#### Create the table

`DB.create_tables([Prediction], safe=True)`

The model that we specified must correspond to a database table.
Creation of these tables is something that is it's own non trivial
headache and this one line of code makes it so that we don't have
to worry about any of it.


## Development

To run it locally, create a virtual environment

```
python3.6 -m venv ./venv
source venv/bin/activate
pip install -r requirements.txt
```

### Example observation

New observation comes in

```
~ > curl -X POST http://localhost:5000/predict -d '{"id": 0, "observation": {"Age": 22.0, "Cabin": null, "Embarked": "S", "Fare": 7.25, "Parch": 0, "Pclass": 3, "Sex": "male", "SibSp": 1}}' -H "Content-Type:application/json"
{
  "id": 1, 
  "observation_id": 0, 
  "predicted_class": true, 
  "proba": 0.09264179297127445, 
  "true_class": null
}
```

Get the true outcome

```
curl -X POST http://localhost:5000/update -d '{"id": 0, "true_class": 1}'  -H "Content-Type:application/json"
```

### Heroku utilities

Push new code after committing it

```
git push heroku master && heroku logs --tail
```

Restart the server

```
heroku ps:restart && heroku logs --tail
```
