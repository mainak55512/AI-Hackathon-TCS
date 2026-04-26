import os
from datetime import timedelta

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = "super-secret-key"  # will use .env later
    JWT_SECRET_KEY = "jwt-secret-key"  # will use .env later
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(basedir, "..", "db", "app.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
