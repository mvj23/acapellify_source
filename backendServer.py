from flask import Flask, request, redirect, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from os.path import join
from os import path as p

from importstatements import *
import Acappellify
import pitch_shifter as shifter
import helperFuncs as utils
import BasesCalc
WINDOW_SIZE = 2048
HOP_SIZE = WINDOW_SIZE // 2

app = Flask(__name__)
CORS(app)

@app.route('/')
def hmm():
    return redirect("https://www.youtube.com/watch?v=dQw4w9WgXcQ")


@app.route('/api/acapellify', methods=['POST'])
def do_an_acapellify():
    if not request.files:
        return "error"
    music = None
    basis = []
    for key, file_storage in request.files.items():
        if key == "music_file":
            music = file_storage
        elif key.find("basis") != -1:
            basis.append(file_storage)

    if not music:
        print("acapellify: missing music")
        return "error"

    if not basis:
        print("acapellify: missing basis")
        return "error"

    return acapellify(music, basis)

def acapellify(music_file, bases):
    CD = p.dirname(p.abspath(__file__))
    temp_directory = CD + "/tmp/"

    if music_file:
        music_filename = secure_filename(music_file.filename)
        path_m = join(temp_directory, music_filename)
        music_file.save(path_m)
    else:
        return "error"
    
    basis_paths_s = []
    basis_paths_n = []
    if bases:
        for file in bases:
            filename = secure_filename(file.filename)
            path = join(temp_directory, filename)
            if filename[0] == "S":
                basis_paths_s.append(path)
            else:
                basis_paths_n.append(path)
            file.save(path)
    else:
        return "error"

    ### Algorithm from main ###

    # Used to be path, just as note
    musicFile, srMusic= librosa.load(path_m, sr = None)

    bases = []
    for next_basis in basis_paths_s:
        next_basis = next_basis.strip()
        basis,srBase = librosa.load(next_basis, sr=None)
        scale_bases_finder = BasesCalc.ScaleBasesCalc(basis_audio=basis, basis_SR=srBase, window_size=WINDOW_SIZE,
                                                          hop_size=HOP_SIZE, graph_it=False)
        bases.extend(scale_bases_finder.get_scale_bases())

    for next_basis in basis_paths_n:
        next_basis = next_basis.strip()
        basis,srBase = librosa.load(next_basis, sr=None)
        bases.append(BasesCalc.ScaleBasesCalc.get_percussive_basis(basis_audio=basis, window_size=WINDOW_SIZE, hop_size=HOP_SIZE))

    bases = shifter.normalize(bases, srBase, WINDOW_SIZE, HOP_SIZE)
    
    music_matrix = utils.matrix_gen(musicFile, srMusic, WINDOW_SIZE, HOP_SIZE)



    numFreqs = music_matrix.shape[0]
    basis_matrix = np.zeros([numFreqs, 0])
    for i, b in enumerate(bases):
        transform = utils.matrix_gen(b, srMusic, WINDOW_SIZE, HOP_SIZE)
        transform = np.average(transform, axis=1) #ask about this
        transform = np.reshape(transform, (transform.shape[0], 1))
        basis_matrix = np.concatenate((basis_matrix, transform[:numFreqs]), axis=1)
        if (TESTING):
            utils.wavwrite("localMusicFiles/bases/base" + str(i) + ".wav", b, srBase)

    A = Acappellify.Acappellify(music_matrix, basis_matrix, musicFile, bases, WINDOW_SIZE, HOP_SIZE)
    acapella = A.createAcappella()
    name_of_output = "Acapellafied" + music_filename
    output_music_file = "tmp\\" + name_of_output + ".wav"
    utils.wavwrite(output_music_file, acapella, srMusic)

    return send_file(output_music_file,
                         mimetype='audio/wav',
                         attachment_filename='result.wav',
                         as_attachment=True)
