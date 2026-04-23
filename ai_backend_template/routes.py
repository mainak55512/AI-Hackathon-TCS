from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required
from werkzeug.security import check_password_hash
from models import User, db
from decorators import role_required

api = Blueprint("api", __name__)


@api.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data.get("username")).first()

    if user and check_password_hash(user.password, data.get("password")):
        token = create_access_token(identity=str(user.id))
        return jsonify(access_token=token)

    return jsonify({"msg": "Bad username or password"}), 401


@api.route("/admin-dashboard", methods=["GET"])
@jwt_required()
@role_required("admin")
def admin_only():
    return jsonify({"msg": "Welcome to the admin panel!"})


@api.route("/main-ui", methods=["GET"])
@jwt_required()
@role_required("user")
def main_ui():
    return jsonify({"msg": "Welcome to the main UI!"})


user_handler = Blueprint("user_handler", __name__)


"""
{
  "roles": [
    "admin",
    "user"
  ],
  "user": "admin_boss"
}"""


@user_handler.route("/roles", methods=["POST"])
@jwt_required()
@role_required("admin")
def get_user_roles():
    data = request.get_json()
    user = User.query.filter_by(username=data.get("username")).first()
    return jsonify({"user": user.username, "roles": [role.name for role in user.roles]})


# TODO:
# Create user API
# Update USer API -> adding/removing roles

# get_user_by_id -> exists -> login, don't exist -> create user
