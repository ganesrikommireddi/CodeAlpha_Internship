from flask import Flask, render_template, request, redirect, flash
from flask_sqlalchemy import SQLAlchemy
import re

app = Flask(__name__)
app.secret_key = "codealpha"

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(15), unique=True, nullable=False)
    city = db.Column(db.String(100), nullable=False)

with app.app_context():
    db.create_all()

@app.route("/")
def home():
    users = User.query.all()
    return render_template("index.html", users=users)

@app.route("/add", methods=["POST"])
def add():
    name = request.form["name"].strip()
    email = request.form["email"].strip().lower()
    phone = request.form["phone"].strip()
    city = request.form["city"].strip()

    if not name or not email or not phone or not city:
        flash("All fields are required!", "danger")
        return redirect("/")

    if not re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email):
        flash("Invalid email!", "danger")
        return redirect("/")

    if not phone.isdigit() or len(phone) != 10:
        flash("Phone number must contain exactly 10 digits!", "danger")
        return redirect("/")

    duplicate = User.query.filter(
        (User.email == email) | (User.phone == phone)
    ).first()

    if duplicate:
        flash("Duplicate record found!", "warning")
        return redirect("/")

    db.session.add(User(name=name, email=email, phone=phone, city=city))
    db.session.commit()
    flash("Record added successfully!", "success")
    return redirect("/")

@app.route("/delete/<int:id>")
def delete(id):
    user = User.query.get_or_404(id)
    db.session.delete(user)
    db.session.commit()
    flash("Record deleted successfully!", "success")
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)
