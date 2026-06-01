from flask import Blueprint
from flask_jwt_extended import jwt_required, get_jwt_identity


extras = Blueprint("extras", __name__)


@extras.route("/hello")
# @jwt_required()
def hello():
    return "Hello from extras.py!"
