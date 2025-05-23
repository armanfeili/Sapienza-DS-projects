from locust import HttpUser, TaskSet, task, between
import random
import json

# Load the questions and options from the JSON data
with open('questionsAndOptions.json', 'r') as f:
    data = json.load(f)
    questions = data[0]['questions']
    options = data[1]['options']
    open_questions = data[2]['open_questions']
    open_question_answers = data[3]['open_question_answers']

class UserBehavior(TaskSet):
    def generate_random_answers(self):
        # Generate random answers
        random_answers = [random.randint(0, len(options) - 1) for _ in questions]
        random_open_answers = [
            random.choice(open_question_answers[f"answers_open_question_{index + 1}"])
            for index in range(len(open_questions))
        ]
        return random_answers, random_open_answers

    def generate_text(self, answers, open_answers):
        question_text = " ".join([
            f"This is {options[answer]} that {questions[index]}"
            if index < 40 else f"This is {options[answer]} that I am {questions[index]}"
            for index, answer in enumerate(answers)
        ])

        open_question_text = " ".join([
            f'For the question "{question}", my answer is "{open_answers[index]}".'
            for index, question in enumerate(open_questions)
        ])

        return f"{question_text} {open_question_text}"

    @task
    def submit_career_recommendations(self):
        answers, open_answers = self.generate_random_answers()
        text = self.generate_text(answers, open_answers)
        self.client.post("/get_career_recommendations", json={"text": text})

class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 5)  # Simulate wait time between tasks

