import os
from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_migrate import Migrate
from os import environ
from dotenv import load_dotenv
import click

# Import the blueprint from api.py
from api import api
from models import db

# Load environment variables from .env file
load_dotenv()

# Initialize the Flask application
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Update "*" to your frontend's domain for better security

# Set the SQLAlchemy database URI
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # This will silence the warning

@app.route('/health')
def health():
    return 'OK', 200

db.init_app(app)
migrate = Migrate(app, db)

app.register_blueprint(api)

# Ensure the app runs on the correct host and port
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)

# Add Flask CLI command for running migrations
@app.cli.command()
def create_db():
    """Creates the database tables."""
    db.create_all()

@app.cli.command()
def drop_db():
    """Drops the database tables."""
    db.drop_all()

@app.cli.command()
def seed_db():
    """Seeds the database with initial data."""
    # Add your seed data here ??
    pass
