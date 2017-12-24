# Scikit binary classifier behind flask app

This is a very simplistic yet effective way to deploy a scikit binary
classifier behind a HTTP server.

## IMPORTANT NOTE

This repo is still in development! You may look for now but until I give the signal, it could change at any moment!

## Development

To run it locally, create a virtual environment

```
python3.6 -m venv ./venv
source venv/bin/activate
pip install -r requirements.txt
```

git push heroku master && heroku logs --tail

### Example observation

New observation comes in

```
 curl -X POST http://localhost:5000/predict -d '{"id": 0, "observation": {"Age": 22.0, "Cabin": null, "Embarked": "S", "Fare": 7.25, "Parch": 0, "Pclass": 3, "Sex": "male", "SibSp": 1}}' -H "Content-Type:application/json"
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
