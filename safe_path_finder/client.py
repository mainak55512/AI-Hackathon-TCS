from flask import jsonify
import json
import requests
from flask_jwt_extended import decode_token
from flask_jwt_extended.exceptions import JWTDecodeError

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os
from datetime import timedelta

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = "super-secret-key"  # will use .env later
    JWT_SECRET_KEY = "jwt-secret-key"  # will use .env later
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(basedir, "token.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)


client = Flask(__name__)
client.config.from_object(Config)
db = SQLAlchemy(client)


# bearer token storing table
class Bearer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    token = db.Column(db.String(500))
    username = db.Column(db.String(80), unique=True)


def update_token(token, username, password, type):
    response = requests.post(
        "http://127.0.0.1:5000/api/login",
        json={"username": username, "password": password},
    )
    if response.status_code == 401:
        print("Incorrect password!")
        exit(1)
    token.token = response.json()["access_token"]
    if type == "new":
        token.username = username
    db.session.add(token)
    db.session.commit()


def login():
    username = input("User name: ")
    password = input("Password: ")
    token = Bearer.query.filter_by(username=username).first()
    if not token:
        token = Bearer()
        update_token(token, username, password, "new")
    else:
        update_token(token, username, password, "existing")

    return username, jsonify({"token": token.token})


def engine():
    """
    Steps:
    ======
    1. Client checks if the stored JWT token is valid,
       if not request new one with username and password
    2. Take start and endpoint from user
    3. Request for shortest path between two points
    """
    [username, token] = login()
    data = json.loads(token.get_data(as_text=True))

    # admin specific func to check available roles:w
    available_roles_of_logged_in_user = requests.post(
        "http://127.0.0.1:5000/user-handler/roles",
        headers={"Authorization": f"Bearer {data['token']}"},
        json={"username": username},
    )

    if available_roles_of_logged_in_user.status_code != 403:
        print(
            "Logged In user roles: ",
            available_roles_of_logged_in_user.json()["roles"],
        )

    start = input("Start: ")
    end = input("End: ")
    headers = {"Authorization": f"Bearer {data['token']}"}
    response = requests.post(
        "http://127.0.0.1:5000/api/get-safest_path",
        json={"start": start, "end": end},
        headers=headers,
    )

    print("\n\n")
    # print("Response Body:", repr(available_roles_of_logged_in_user.text))
    print(response.text)


if __name__ == "__main__":
    with client.app_context():
        db.create_all()
        engine()
