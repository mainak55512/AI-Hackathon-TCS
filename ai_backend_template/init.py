from app import app, db
from models import Role, User
from werkzeug.security import generate_password_hash


def init_db():
    with app.app_context():
        db.create_all()

        role_names = ["admin", "user"]
        roles_in_db = {}

        for name in role_names:
            role = Role.query.filter_by(name=name).first()
            if not role:
                role = Role(name=name)
                db.session.add(role)
                db.session.commit()
                print(f"Created role: {name}")
            roles_in_db[name] = role

        # Admin
        if not User.query.filter_by(username="admin_boss").first():
            hashed_pw = generate_password_hash("password123")
            admin_user = User(username="admin_boss", password=hashed_pw)
            admin_user.roles.append(roles_in_db["admin"])
            db.session.add(admin_user)
            print("Created admin_boss")

        # Normal user
        if not User.query.filter_by(username="poor_user").first():
            hashed_pw = generate_password_hash("password456")
            norm_user = User(username="poor_user", password=hashed_pw)
            norm_user.roles.append(roles_in_db["user"])
            db.session.add(norm_user)
            print("Created poor_user")

        db.session.commit()
        print("Database initialization complete!")


if __name__ == "__main__":
    init_db()
