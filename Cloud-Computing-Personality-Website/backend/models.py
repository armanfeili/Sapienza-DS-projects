from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class PersonalityRecord(db.Model):
    __tablename__ = 'personality_records'
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    result_mbti = db.Column(db.Text, nullable=False)
    result_big_five = db.Column(db.Text, nullable=False)
    career_data = db.Column(db.Text, nullable=False)

    def json(self):
        return {
            'id': self.id,
            'text': self.text,
            'result_mbti': self.result_mbti,
            'result_big_five': self.result_big_five,
            'career_data': self.career_data
        }
