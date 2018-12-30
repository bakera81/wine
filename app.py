from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from spinewectator import tweet

app = Flask(__name__)


scheduler = BackgroundScheduler()
job = scheduler.add_job(tweet, 'interval', hours=2)
scheduler.start()

@app.route('/health')
def health():
    return 'Hallo! (2)'
