# Scikit model behind flask app on heroku

## tl;dr

You can deploy your own model by

1. Copying the contents of this repo to a new directory
1. Replace `pipeline.pickle`, `dtypes.pickle`, and `columns.json` with
   your own.
1. [Deploy to heroku](https://github.com/LDSSA/heroku-model-deploy#deploy-to-heroku)

You'll probably run into a few issues along the way which is why you'll at least want to
skim the contents of the notebooks and this README so you can at least have an idea of
where to look when you hit a bump in the road.

## Intro

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

## Before continuing

Topic #1 is the only one that is not covered here in this readme. It is covered in two notebooks
that you must read before moving on with the rest of this README.

[Notebook #1](https://github.com/LDSSA/heroku-example/blob/master/Train%20and%20Serialize.ipynb) has
to do with training and serializing a scikit model as well as how to prepare a new observation
that arrives for prediction.

[Notebook #2](https://github.com/LDSSA/heroku-example/blob/master/Deserialize%20and%20use.ipynb) has
to do with deserialization so that you can re-use a model on new observations without having to re-train
it.

## Conda environments

We use conda in the exact same way as in the [batch3-students repo](https://github.com/LDSSA/batch3-students).

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

#### Get a project started

In order to use flask, you will need to be writing some code in a regular
python file - no more notebooks here. The first step (assuming you have already
created an environment with `environment.yml`) is to import it at the top of the file. 
Let's pretend that we are working in a file called `app.py` in our newly created 
conda environment.

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
This will allow us to use a local database called [sqlite](https://en.wikipedia.org/wiki/SQLite) (which is basically a file)
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

## Integrate data model with webserver

Now that we have a webserver and a data model that we are happy
with, the next question is how do we put them together? It's
actually pretty straightforward!

```py
import json
import pickle
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict


########################################
# Begin database stuff

DB = SqliteDatabase('predictions.db')


class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model


with open('columns.json') as fh:
    columns = json.load(fh)


with open('pipeline.pickle', 'rb') as fh:
    pipeline = pickle.load(fh)


with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################


########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    obs_dict = request.get_json()
    _id = obs_dict['id']
    observation = obs_dict['observation']
    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    proba = pipeline.predict_proba(obs)[0, 1]
    response = {'proba': proba}
    p = Prediction(
        observation_id=_id,
        proba=proba,
        observation=request.data,
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = "ERROR: Observation ID: '{}' already exists".format(_id)
        response["error"] = error_msg
        print(error_msg)
        DB.rollback()
    return jsonify(response)


# End webserver stuff
########################################

if __name__ == "__main__":
    app.run(debug=True)

```

One piece of the code above that might not be clear at first is:

```py
    p = Prediction(
        observation_id=_id,
        proba=proba,
        observation=request.data,
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = "ERROR: Observation ID: '{}' already exists".format(_id)
        response["error"] = error_msg
        print(error_msg)
        DB.rollback()
```

What is this code doing?. When we receive a new prediction request, we want to store such request
in our database (to keep track of our model performance). With peewee, we save a new Prediction (basically
a new row in our table) with the `save()` method, which is very neat and convenient.

However, because our table has a unique constraint (no two rows can have the same `observation_id` is a unique field),
if we perform the same prediction request twice (with the same id) the system will crash because pewee can't save
again an already saved observation_id, and it will throw an `IntegrityError` (as in, we would be asking pewee to violate
the integrity of the table unique id requirement if we saved a duplicated id, right?).

To avoid that we do a simple try/Except block, we make sure we are only catching the integrity error, then, for those cases
where we try a request with the same observation_id, we print a nice error message an we do a database rollback (to close the current save transaction that has failed).

Once your app is setup like this, you can test this with the following command:

```bash
~ > curl -X POST http://localhost:5000/predict -d '{"id": 0, "observation": {"Age": 22.0, "Cabin": null, "Embarked": "S", "Fare": 7.25, "Parch": 0, "Pclass": 3, "Sex": "male", "SibSp": 1}}' -H "Content-Type:application/json"
{
  "proba": 0.09264179297127445
}
```

Now let's take note of the few things that changed

1. The structure of the json input changed. It now includes to top level entries:
    - `id` - This is the unique identifier of the observation
    - `observation` - This is the actual observation contents that will be sent through
      the pipeline we have un-pickled.
1. We create an instance of `Prediction` with the 3 fields that we care about
1. We call `save()` on the prediction to save it to the database
1. We return `proba` so that the caller of the HTTP endpoint knows what you are
saying about the observation.

## Receiving updates

Now that we have a way to provide prediction AND keep track of them, we should
take it to the next level and provide ourselves with a way to receive updates
on observations that we have judged with our predictive model.

We can do this with one extra endpoint that is very straightforward and only
introduced one new concept of database querying through the ORM.

```py
@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['id'])
        p.true_class = obs['true_class']
        p.save()
        return jsonify(model_to_dict(p))
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['id'])
        return jsonify({'error': error_msg})
```

Assuming that we have already processed an observation with id=0, we
can now recieve and record the true outcome. Imagine that it is discovered
later on that the person with id=0 didn't survive the titanic disaster. They
would probably enter something into a content management system that
would then trigger a call to your server that would end up looking like
the following:

```bash
~ > curl -X POST http://localhost:5000/update -d '{"id": 0, "true_class": 1}'  -H "Content-Type:application/json"
{
  "id": 1,
  "observation": "{\"id\": 0, \"observation\": {\"Age\": 22.0, \"Cabin\": null, \"Embarked\": \"S\", \"Fare\": 7.25, \"Parch\": 0, \"Pclass\": 3, \"Sex\": \"male\", \"SibSp\": 1}}",
  "observation_id": 0,
  "proba": 0.09264179297127445,
  "true_class": 1
}
```

Similarly to when we saved the prediction requests, we validate that the observation_id we want to update actually exists.

Now to wrap it all up, the way that we can interpret this sequence of events is the following:

1. We provided a prediction of 0.092 probability of survival
1. We found out later that the person didn't survive

## Deploy to Heroku

It's cool and all that we can run the servers on our own machines. However, it doesn't
do much good in terms of making the model available to the rest of the world. All this
`localhost` stuff doesn't help anybody that's not typing on your local machine.

So let's take all of the work we've done getting this running and put it on heroku
where it can generate real business value. For this part, you can use any server
that has a static IP address though since we want to avoid the overhead of administering
our own server, we will use a servive to do this for us called [heroku](https://www.heroku.com/)
This is one of the oldest managed platforms out there and is quite robust, well-known, and
documented. However, be careful before you move forward with a big project o Heroku -
it can get CRAZY expensive REALLY fast.

However, for our purposes, they offer a free tier webserver and database that is enough to suit our
needs and we can do deployments in a few commands super easily. This may be a bit tough
for some of you but trust me: the alternative of admining your own server is MUCH more difficult.

### Sign up and set up at heroku

Go to the [signup page](https://signup.heroku.com/) and register for the free tier.

Once this is all done, go to the [dashboard](https://dashboard.heroku.com/apps) and create a new
app:

![create new app](https://i.imgur.com/SYyFMV1.png)

Then on the next screen, give it a name and make sure that it's in the Europe zone. It won't
kill anobody to have it in the land of the free but it's kinda far...

![select name and region](https://i.imgur.com/oUPNzOk.png)

Once this is done, select "create app" and you'll be sent to a page that's a bit intimidating
beacuse it just has a lot of stuff. Don't worry though, it's pretty simple what we need
to do next.

First up, make sure that you select the Heroku Git deployment method. It should already be selected
so I don't think you'll need to do anything.

![heroku git](https://i.imgur.com/xt0dAhq.png)

One last bit is missing here: the database. We are going to use a big boy database
called postgresql and luckily heroku has a free tier that allows you to store
up to 10,000 entries which is enough for our purposes (this means that you should try to be conservative
with how you connect to the app and dont go crazy with it, if the database gets full your app will stop working!, you can check heroku's postgresql guide [here](https://devcenter.heroku.com/articles/heroku-postgresql)).

To add the database, navigate to `Resources` and search for `postgres`, then select `Heroku Postgres` and the
`Hobby dev - free` tier:

![add postgres](https://i.imgur.com/rZvNnuB.png)

### Now lets deploy the titanic model

Let's deploy the server that's contained in this repository. The code is in `app.py` and
there's a few other files that are required but we'll go over those a bit later.

First step toward deployment is to make sure that this repo is cloned on your local
machine.

Once this is done, you will want to download and install the
[heroku cli](https://devcenter.heroku.com/articles/heroku-cli).

After the heroku cli is installed, you'll need to open a command prompt and
log in. You will use the same credentials that you use to log in through the
web interface with and it should look something like the following, part of
which is asking you to open up a browser and log in. Should be pretty
straightforward.

```bash
~ > heroku login
heroku: Press any key to open up the browser to login or q to exit:
Opening browser to https://cli-auth.heroku.com/auth/browser/.........
Logging in... done
Logged in as sam@puppiesarecute.com
```

Great! now when you execute commands on your local machine, the heroku cli will know
who you are!

Now you will want to navigate on the command line to the location of the folder in which
you cloned the repository. It should look something like this:

```bash
~ > cd ldssa/heroku-model-deploy/
~ > ls
Deserialize and use.ipynb	README.md			columns.json			requirements.txt
LICENSE				Train and Serialize.ipynb	dtypes.pickle			titanic.csv
Procfile			app.py                      pipeline.pickle
```

And make sure that heroku knows about the app you just created by adding a git
remote by executing the following command but replacing "heroku-model-deploy"
with the name of the app you just created:

```bash
~ > heroku git:remote -a batch3-capstone-demo
set git remote heroku to https://git.heroku.com/batch3-capstone-demo.git
```

At this point, we'll need to do something a bit extra in order to be able to use
our conda environment by specifying and configuring a heroku buildpack:

```
~ > heroku stack:set container
Stack set. Next release on ⬢ batch3-capstone-demo will use container.
Run git push heroku master to create a new release on ⬢ batch3-capstone-demo.
```

Now we can push to heroku and our app will be deployed **IN THE CLOUD, WAAAT?!?!** It is important to remember, only
the changes that you have commited (with git add & git commit) will be deployed. So if you
change your pipeline and retrain a different model you need to commit the changes before
pushing to heroku).

```bash
~ > git push heroku master
Enumerating objects: 5, done.
Counting objects: 100% (5/5), done.
Delta compression using up to 4 threads
Compressing objects: 100% (3/3), done.
Writing objects: 100% (3/3), 302 bytes | 302.00 KiB/s, done.
Total 3 (delta 2), reused 0 (delta 0)
remote: Compressing source files... done.
remote: Building source:
remote: === Fetching app code
remote:
remote: === Building web (Dockerfile)
remote: Sending build context to Docker daemon  543.7kB
remote: Step 1/6 : FROM continuumio/miniconda3
remote: latest: Pulling from continuumio/miniconda3
remote: b8f262c62ec6: Pulling fs layer
remote: 0a43c0154f16: Pulling fs layer
remote: 906d7b5da8fb: Pulling fs layer
remote: 906d7b5da8fb: Verifying Checksum
remote: 906d7b5da8fb: Download complete
remote: b8f262c62ec6: Verifying Checksum
remote: b8f262c62ec6: Download complete
remote: 0a43c0154f16: Verifying Checksum
remote: 0a43c0154f16: Download complete

...

remote: 61c68312a080: Pushed
remote: latest: digest: sha256:b094fd516873cea432cd199f4371d3e190bb06bbbcb436ba882d3eacc591e327 size: 1375
remote:
remote: Verifying deploy... done.
To https://git.heroku.com/batch3-capstone-demo.git
   8eb8b66..05ae35d  master -> master
```
 And boom! We're done and deployed! You can actually see this working by executing
 some of the curl commands that we saw before but using `https://<your-app-name>.herokuapp.com`
 rather than `http://localhost` like we saw earlier. For my app it looks like the following:

 ```
 ~ > curl -X POST https://batch3-capstone-demo.herokuapp.com/predict -d '{"id": 0, "observation": {"Age": 22.0, "Cabin": null, "Embarked": "S", "Fare": 7.25, "Parch": 0, "Pclass": 3, "Sex": "male", "SibSp": 1}}' -H "Content-Type:application/json"
{
  "proba": 0.09264179297127445
}
 ```

 And we can recieve updates like the following:

 ```bash
~ > curl -X POST https://batch3-capstone-demo.herokuapp.com/update -d '{"id": 0, "true_class": 1}' -H "Content-Type:application/json"
{
  "id": 1,
  "observation": "{\"id\": 0, \"observation\": {\"Age\": 22.0, \"Cabin\": null, \"Embarked\": \"S\", \"Fare\": 7.25, \"Parch\": 0, \"Pclass\": 3, \"Sex\": \"male\", \"SibSp\": 1}}",
  "observation_id": 0,
  "proba": 0.0926418,
  "true_class": 1
}
```

How does all of this work you ask? Well it has to do with the fact
that there exists a `Dockerfile` and `heroku.yml` files in this repo.
Just be sure that both of these are in your repo and that they are
committed before you `heroku push` and everything should work. If you
do a deploy and it doesn't work, take a look at the files and see
if you might have done something such as change a filename or something
else that would break the boilerplate assumptions that we've made here.

You can see the logs (which is helpful for debugging) with the `heroku logs` command.
Here are the logs for the two calls we just made:

```
batch3-capstone-demo master > heroku logs -n 5
2017-12-27T20:14:59.351793+00:00 app[web.1]: [2017-12-27 20:14:59 +0000] [4] [INFO] Using worker: sync
2017-12-27T20:14:59.359149+00:00 app[web.1]: [2017-12-27 20:14:59 +0000] [8] [INFO] Booting worker with pid: 8
2017-12-27T20:14:59.371891+00:00 app[web.1]: [2017-12-27 20:14:59 +0000] [9] [INFO] Booting worker with pid: 9
2017-12-27T20:15:00.678404+00:00 heroku[web.1]: State changed from starting to up
2017-12-27T20:19:25.944435+00:00 heroku[router]: at=info method=POST path="/predict" host=batch3-capstone-demo.herokuapp.com request_id=79138602-5b95-497a-9b69-c2528a2bbfc9 fwd="86.166.46.98" dyno=web.1 connect=0ms service=496ms status=200 bytes=187 protocol=https
2017-12-27T20:20:46.033529+00:00 heroku[router]: at=info method=POST path="/update" host=batch3-capstone-demo.herokuapp.com request_id=cc92e857-895d-425b-ab00-a92862e1253e fwd="86.166.46.98" dyno=web.1 connect=1ms service=9ms status=200 bytes=417 protocol=https
```

### Import problems

If you are using something like a custom transformer, and getting an import error having to do with your custom code
when unpickling, you'll need to do the following:

#### Put your custom code in a package

Let's say that you have a custom transformer called `MyCustomTransformer` that is part of your
pickled pipeline. In that case, you'll want to create a [python package](https://www.learnpython.org/en/Modules_and_Packages)
from which you import in both your traning and deployment code.

In our example, let's create the following package called `custom_transformers` by just creating a directory
with the same name and putting two files inside of it so that it looks like this:

```
└── custom_transformers
    ├── __init__.py
    └── transformer.py
```

And inside of `transformer.py` you can put the code for `MyCustomTransformer`. Then in both your training code as
well as your `app.py`, you can import them with:

```
from custom_transformers.transformer import MyCustomTransformer
```

#### Create a setup.py

Then you will need to put a `setup.py` in the root of the project directory that should look like this:

```
from setuptools import setup
from os import path


setup(
    name='my-custom-transformers',
    version='0.0.1',
    description='heroku model deploy',
    url='https://nuna.biz',
    author='Some data scientist',
    author_email='nuna@nuna.biz',
    packages=['custom_transformers'],
    install_requires=[],
)
```

What this does it make your package `pip` installable. You may or may not need to
install it in your environment by executing `pip install -e .` in your current directory.

#### Include it in your environment.yml

Since we are using anaconda to satisfy dependencies and we have created a package
that is `pip` installable, we're in a bit of a pickle that requires one little
hackzinho to make it work. We need to tell anaconda to interpret the current package
as a pip package rather than an anaconda dependency because anaconda isn't smart
enough to work with a `setup.py`:

```
channels:
  - conda-forge
  - defaults
dependencies:
  ...
  - pip
  - pip:
    - -e .
```

Now when you push to heroku, the environment should be able to find your 
custom code when unpickling.

### Last few notes

There were are few additional changes to `app.py` and the rest of the repo that we haven't covered yet so
let's get that out of the way. You probably won't need to know much about them but if you are having
troubleshooting issues, knowing the following may come in handy.

#### The db connector

Instead of having just an sqlite connector, we needed to add another block of code that would
detect if it is being run on heroku or not and if so, connect to the postgresql database. We know
that we are on heroku if there is a `DATABASE_URL` environment variable. When we add the postgres database in heroku, heroku will
automatically add an environment variable (DATABASE_URL) with the conexion we need.

```py
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
```

#### The procfile

The heroku [Procfile](https://devcenter.heroku.com/articles/procfile) is how
we tell heroku to use the code we have deployed to it. The contents of ours
is very simple and tells [gunicorn](http://gunicorn.org/) that there's an `app.py`
file and inside of that file, theres an object called `app` that contains a
[wsgi](https://en.wikipedia.org/wiki/Web_Server_Gateway_Interface) server that it
can use to listen for incoming connections:

```
web: gunicorn app:app
```

### Heroku useful snippets

Push new code after committing it

```bash
git push heroku master && heroku logs --tail
```

Restart the server

```bash
heroku ps:restart && heroku logs --tail
```

Check the latest 300 logs of your application

```bash
heroku logs -n 300
```
