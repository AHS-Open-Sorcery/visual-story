from flask import *
from blacklist import blacklisted
from werkzeug.utils import secure_filename
from ObjectIdentification import imageDetection
from story import prompt_form, generation
from collections import Counter
from threading import Lock
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
        return redirect(url_for('progress', filename=filename))


lock = Lock()

@app.route('/progress')
def progress():
    lock.acquire()
    filename = request.args.get('filename')
    if filename is None:
        return 'failed; filename none', 422

    # step 1: identify objects
    objects = list(imageDetection.getObjects(os.path.join('images', filename)))
    print(objects)
    # occupations = list(imageDetection.getOccupation(os.path.join('images', filename)))
    # print(occupations)

    # step 2: build prompt
    prompt = prompt_form.prompt(Counter(list(map(lambda tup: tup[0], objects))))

    # step 3: send prompt to desired AI
    generated = generation.generate(prompt)

    # step 4: retrieve generated story
    # step 5: profit
    lock.release()
    return generated


if __name__ == '__main__':
    app.run()
