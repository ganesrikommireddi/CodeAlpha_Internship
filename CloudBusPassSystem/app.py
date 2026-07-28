from flask import Flask, render_template, request, redirect, session, flash
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.secret_key = "buspass"

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

class BusPass(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    source = db.Column(db.String(100))
    destination = db.Column(db.String(100))

with app.app_context():
    db.create_all()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        if User.query.filter_by(email=request.form["email"]).first():
            flash("Email already registered.")
            return redirect("/register")
        user = User(
            name=request.form["name"],
            email=request.form["email"],
            password=request.form["password"]
        )
        db.session.add(user)
        db.session.commit()
        flash("Registration successful!")
        return redirect("/login")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(
            email=request.form["email"],
            password=request.form["password"]
        ).first()
        if user:
            session["user_id"] = user.id
            session["user"] = user.name
            return redirect("/dashboard")
        flash("Invalid login")
    return render_template("login.html")

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user_id" not in session:
        return redirect("/login")
    if request.method == "POST":
        new_pass = BusPass(
            name=session["user"],
            source=request.form["source"],
            destination=request.form["destination"]
        )
        db.session.add(new_pass)
        db.session.commit()
        return redirect(f"/pass/{new_pass.id}")
    return render_template("dashboard.html")

@app.route("/pass/<int:pass_id>")
def pass_page(pass_id):
    if "user_id" not in session:
        return redirect("/login")
    data = BusPass.query.get_or_404(pass_id)
    return render_template("pass.html", data=data)

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)
