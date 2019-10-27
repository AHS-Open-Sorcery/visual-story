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


task = None
generated = None


def run_process(params, out: Queue):
    filename, tags = params

    try:
        # step 1: identify objects
        objects_confidence = list(imageDetection.getObjects(os.path.join('images', filename)))
        objects = list(map(lambda tup: tup[0], objects_confidence))
        if len(objects) == 0:
            out.put(ProgressUpdate(1, 'no objects found', fail=True))
            return

        out.put(ProgressUpdate(0.27, 'detected ' + ', '.join(objects)))

        # TODO come back to occupation detection
        # occupations = list(imageDetection.getOccupation(os.path.join('images', filename)))
        # print(occupations)

        # step 2: build prompt
        counter = Counter(objects)
        if counter['person'] == 1:
            occupation = imageDetection.getOccupation(os.path.join('images', filename))
            counter.pop('person')
            counter[occupation] += 1

        prompt = prompt_form.prompt(counter, tags)

        # step 3: send prompt to desired AI
        global generated
        for i in range(3):
            generated = prompt + generation.generate(prompt, out)
            if blacklisted(generated):
                out.put(ProgressUpdate(0.33, 'found obscene words! retrying'))
            else:
                break
    except Exception as e:
        out.put(ProgressUpdate(1, str(e), fail=True))


@app.route('/begin')
def begin():
    global task
    if task is not None and task.running():
        return 'somebody else is using the gpu. hold on', 400

    filename = request.args.get('filename')
    tags = request.args.get('tags')
    if tags is None:
        tags = []
    else:
        tags = tags.split(',')

    if filename is None:
        return 'failed; filename none', 422

    task = Task(run_process, (filename, tags))
    return 'yes'


@app.route('/progress')
def progress():
    if task is None:
        return json.dumps({'status': 'fail', 'message': 'no current task'}), 400
    update: ProgressUpdate = task.read_progress()
    return json.dumps({'status': 'success' if not update.fail else 'fail',
                       'message': update.message, 'progress': update.progress})


@app.route('/result')
def result():
    if generated is None:
        return 'no results', 400
    else:
        return generated


if __name__ == '__main__':
    app.run(host='0.0.0.0')
