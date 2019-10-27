from queue import Queue

from flask import *
from blacklist import blacklisted
from werkzeug.utils import secure_filename
from ObjectIdentification import imageDetection
from story import prompt_form, generation
from collections import Counter
from threading import Lock
from task import Task, ProgressUpdate
import json

import os
app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('upload.html')


@app.route('/submit', methods=['POST'])
def submit():
    if 'file' not in request.files:
        return 'missing files', 422
    file = request.files['file']

    if file.filename == '':
        return 'missing files', 422

    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join('images', filename))
        return redirect(url_for('begin', filename=filename, tags=request.form.get('tags')))


lock = Lock()
task = None


def run_process(params, out: Queue):
    lock.acquire()
    filename, tags = params

    # step 1: identify objects
    objects_confidence = list(imageDetection.getObjects(os.path.join('images', filename)))
    objects = list(map(lambda tup: tup[0], objects_confidence))
    print(objects)
    out.put(ProgressUpdate(0.27, 'detected ' + ', '.join(objects)))

    # TODO come back to occupation detection
    # occupations = list(imageDetection.getOccupation(os.path.join('images', filename)))
    # print(occupations)

    # step 2: build prompt
    prompt = prompt_form.prompt(Counter(objects), tags)
    out.put(ProgressUpdate(0.3, 'built prompt for GPT-2'))

    # step 3: send prompt to desired AI
    generated = generation.generate(prompt, out)
    lock.release()
    return generated


@app.route('/begin')
def begin():
    if lock.locked():
        return 'duplicate req', 400

    filename = request.args.get('filename')
    tags = request.args.get('tags')
    if tags is None:
        tags = []
    else:
        tags = tags.split(',')

    if filename is None:
        return 'failed; filename none', 422

    global task
    task = Task(run_process, (filename, tags))
    return 'yes'


@app.route('/progress')
def progress():
    if task is None:
        return json.dumps({'status': 'fail', 'message': 'no current task'}), 400
    update: ProgressUpdate = task.read_progress()
    return json.dumps({'status': 'success', 'message': update.message, 'progress': update.progress})


if __name__ == '__main__':
    app.run()
