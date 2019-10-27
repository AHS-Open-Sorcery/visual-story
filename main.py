from flask import *
from blacklist import blacklisted
from werkzeug.utils import secure_filename
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


@app.route('/progress')
def progress():
    filename = request.args.get('filename')
    if filename is None:
        return 'failed; filename none', 422

    # step 1: identify objects
    # step 2: build prompt
    # step 3: send prompt to desired AI
    # step 4: retrieve generated story
    # step 5: profit
    
    print(filename)
    return 'yeah'


if __name__ == '__main__':
    app.run()
