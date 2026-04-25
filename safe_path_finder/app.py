from flask import Flask
from flask_jwt_extended import JWTManager
from config import Config
from models import db
from routes import api, user_handler

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)
jwt = JWTManager(app)

app.register_blueprint(api, url_prefix="/api")
app.register_blueprint(user_handler, url_prefix="/user-handler")

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
