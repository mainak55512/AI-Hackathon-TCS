from functools import wraps
from flask import jsonify
from flask_jwt_extended import get_jwt_identity, verify_jwt_in_request
from models import User


def role_required(required_role):
    def wrapper(fn):
        @wraps(fn)
        def decorator(*args, **kwargs):
            verify_jwt_in_request()
            user_id = get_jwt_identity()

            user = User.query.get(user_id)

            if not user:
                return jsonify({"msg": "User not found"}), 404

            roles = [r.name for r in user.roles]
            if required_role not in roles:
                return jsonify(
                    {"msg": f"Forbidden: Requires {required_role} role"}
                ), 403

            return fn(*args, **kwargs)

        return decorator

    return wrapper
