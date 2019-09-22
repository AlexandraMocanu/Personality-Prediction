import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import pickle
import random
import glob
from AmigosModel import AmigosModel
from random_word import RandomWords
from flask import Flask, Response, jsonify, flash, render_template, request, redirect, url_for,\
    send_from_directory, session, escape
from werkzeug.utils import secure_filename
from camera import VideoCamera

app = Flask(__name__)
app.secret_key = os.urandom(16)

# for UPLOAD
UPLOAD_FOLDER = './uploads'
RECORDS_FOLDER = './records'
INITIAL_DATA_FOLDER = './initial_data' # if you change this, also change the recorder.js
ALLOWED_EXTENSIONS = set(['avi', 'mov', 'mp4', 'png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RECORDS_FOLDER'] = RECORDS_FOLDER
app.config['INITIAL_DATA_FOLDER'] = INITIAL_DATA_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024 #500 mb
app.config['USERNAME'] = ""

# for RECORDING
video_camera = None
global_frame = None

# recording source code
# https://github.com/yushulx/web-camera-recorder

MODEL = AmigosModel()

def init():
    r = RandomWords()
    user_key = random.randint(1,10000)
    session['username'] = (r.get_random_word(hasDictionaryDef="true", minLength=5, maxLength=15)+str(user_key))
    app.config['USERNAME'] = session['username']

    MODEL.set_username(username=app.config['USERNAME'])

    # print(os.path.join(app.instance_path))


# home page
@app.route("/")
@app.route("/home", methods=['GET', 'POST'])
def home():
    # init session if it was not
    if not 'username' in session:
        init()

    # create folders
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.mkdir(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['RECORDS_FOLDER']):
        os.mkdir(app.config['RECORDS_FOLDER'])
    if not os.path.exists(app.config['INITIAL_DATA_FOLDER']):
        os.mkdir(app.config['INITIAL_DATA_FOLDER'])
    
    return render_template('home.html')


# UPLOAD page
@app.route("/upload", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file_4upload' in request.files:
            file = request.files['file_4upload']
            # if user does not select file, browser also
            # submit a empty part without filename
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filename_extension = filename.split('.')[1]
                filename = filename.split('.')[0]
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], "upload_"+app.config['USERNAME']+"."+filename_extension))
                # uploaded_file - function name
                # return redirect(url_for('uploaded_file', filename=filename+app.config['USERNAME']))
                return predict(filename="upload_"+app.config['USERNAME']+"."+filename_extension)
        elif 'save_selfassess' in request.form:
            with open(os.path.join(app.config['INITIAL_DATA_FOLDER'], 'save_selfassess_'+app.config['USERNAME']+'.json'), 'w') as f:
                json.dump(request.form, f)
            return render_template('upload.html', saved_assess=True)
        else:
            flash('No file upload or self asessment!')
            return redirect(request.url)

    return render_template('upload.html', saved_assess=False)


# test for uploaded file to be valid
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# watch for uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


# RECORD page
@app.route('/record', methods=['GET', 'POST'])
def record():
    if 'save_selfassess' in request.form:
        with open(os.path.join(app.config['INITIAL_DATA_FOLDER'], 'save_selfassess_'+app.config['USERNAME']+'.json'), 'w') as f:
            json.dump(request.form, f)
        return render_template('record.html', saved_assess=True, \
            username=app.config['USERNAME'], records_folder=app.config['RECORDS_FOLDER'])

    return render_template('record.html', saved_assess=False, \
        username=app.config['USERNAME'], records_folder=app.config['RECORDS_FOLDER'])


# recording part
@app.route('/record/record_status', methods=['POST'])
def record_status():
    global video_camera 
    if video_camera == None:
        video_camera = VideoCamera(app.config['USERNAME'], app.config['RECORDS_FOLDER'])

    json = request.get_json()

    status = json['status']

    if status == "true":
        video_camera.start_record()
        return jsonify(result="started")
    else:
        video_camera.stop_record()
        return jsonify(result="stopped")


# video capture
def video_stream():
    global video_camera 
    global global_frame

    if video_camera == None:
        video_camera = VideoCamera(app.config['USERNAME'], app.config['RECORDS_FOLDER'])
        
    while True:
        frame = video_camera.get_frame()

        if frame != None:
            global_frame = frame
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')


# record show page
@app.route('/record/video_viewer')
def video_viewer():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predict')
def predict(filename="record"):
    
    initial_data_usr = glob.glob(os.path.join(app.config['INITIAL_DATA_FOLDER'], 'save_selfassess_'+app.config['USERNAME']+'.json'))

    if filename == "record":
        print("\nProcessing recording ...")
        records_usr = glob.glob(os.path.join(app.config['RECORDS_FOLDER'], 'video_'+app.config['USERNAME']+'.avi'))\

        if not records_usr == []: # user recorded a video
            # amigos_model = AmigosModel(username=app.config['USERNAME'], input_type='video', \
            #     initial_data=initial_data_usr, records=records_usr[0])
            MODEL.set_config(input_type='video')
            MODEL.load_initial_data(initial_data=initial_data_usr)
            MODEL.set_prediction_data(records=records_usr[0])
            filename=records_usr[0]
        else:
            return render_template('error.html', username=app.config['USERNAME'])
    else:
        uploads_usr = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], 'upload_'+app.config['USERNAME']+'*'))

        if not uploads_usr == []:
            if '.png' in filename or '.jpg' in filename or '.jpeg' in filename:
                # amigos_model = AmigosModel(username=app.config['USERNAME'], input_type='image', \
                #     initial_data=initial_data_usr, image=uploads_usr[0])
                print("\nProcessing image ...")
                MODEL.set_config(input_type='image')
                MODEL.load_initial_data(initial_data=initial_data_usr)
                MODEL.set_prediction_data(image=uploads_usr[0])
            elif '.avi' in filename or '.mov' in filename or '.mp4' in filename: # a video
                # amigos_model = AmigosModel(username=app.config['USERNAME'], input_type='video', \
                #     initial_data=initial_data_usr, video=uploads_usr[0])
                print("\nProcessing video ...")
                MODEL.set_config(input_type='video')
                MODEL.load_initial_data(initial_data=initial_data_usr)
                MODEL.set_prediction_data(video=uploads_usr[0])
            
            filename=uploads_usr[0]
        else:
            return render_template('error.html', username=app.config['USERNAME'])

    also_lrcn = False
    predictions = MODEL.get_predictions()

    panas_conv2d = {}
    personality_conv2d = {}
    fassess_conv2d = {}
    panas_lrcn = {}
    personality_lrcn = {}
    fassess_lrcn = {}

    if not predictions['pred_lrcn'] == {}: #only conv2d
        lrcn = predictions['pred_lrcn']
        also_lrcn = True

        panas_lrcn = {
            'Interested': lrcn['panas_interested'], 'Excited': lrcn['panas_excited'], 'Strong': lrcn['panas_strong'],
            'Distressed': lrcn['panas_distressed'], 'Upset': lrcn['panas_upset'], 'Guilty': lrcn['panas_guilty'],
            'Enthusiastic': lrcn['panas_enthusiastic'], 'Proud': lrcn['panas_proud'], 'Alert': lrcn['panas_alert'],
            'Inspired': lrcn['panas_inspired'], 'Scared': lrcn['panas_scared'], 'Hostile': lrcn['panas_hostile'],
            'Irritable': lrcn['panas_irritable'], 'Ashamed': lrcn['panas_ashamed'], 'Determined': lrcn['panas_determined'],
            'Attentive': lrcn['panas_attentive'], 'Active': lrcn['panas_active'], 'Nervous': lrcn['panas_nervous'],
            'Jittery': lrcn['panas_jittery'], 'Afraid': lrcn['panas_afraid']
        }

        personality_lrcn = {
            'Open': lrcn['personality_open'],
            'Warmhearted': lrcn['personality_warmhearted'], 'Extroverted': lrcn['personality_extroverted'],
            'Exuberant': lrcn['personality_exuberant'], 'Vivacious': lrcn['personality_vivacious'], 'Inward_looking': lrcn['personality_inward_looking'],
            'Introverted': lrcn['personality_introverted'], 'Reserved': lrcn['personality_reserved'], 'Silent': lrcn['personality_silent'],
            'Shy': lrcn['personality_shy'], 'Altruistic': lrcn['personality_altruistic'], 'Sympathetic': lrcn['personality_sympathetic'],
            'Agreeable': lrcn['personality_agreeable'], 'Generous': lrcn['personality_generous'], 'Hospitable': lrcn['personality_hospitable'],
            'Cynical': lrcn['personality_cynical'], 'Egocentric': lrcn['personality_egocentric'], 'Egoistic': lrcn['personality_egoistic'],
            'Suspicious': lrcn['personality_suspicious'], 'Revengeful': lrcn['personality_revengeful'], 'Conscientious': lrcn['personality_conscientious'],
            'Diligent': lrcn['personality_diligent'], 'Methodical': lrcn['personality_methodical'], 'Orderly': lrcn['personality_orderly'],
            'Precise': lrcn['personality_precise'], 'Untidy': lrcn['personality_untidy'], 'Careless': lrcn['personality_careless'],
            'Rash': lrcn['personality_rash'], 'Inconstant': lrcn['personality_inconstant'], 'Heedless': lrcn['personality_heedless'],
            'Calm': lrcn['personality_calm'], 'Impassive': lrcn['personality_impassive'], 'Serene': lrcn['personality_serene'],
            'Self_assured': lrcn['personality_self_assured'], 'Anxious': lrcn['personality_anxious'], 'Emotional': lrcn['personality_emotional'],
            'Jealous': lrcn['personality_jealous'], 'Nervous_P': lrcn['personality_nervous_P'], 'Touchy': lrcn['personality_touchy'],
            'Susceptible': lrcn['personality_susceptible'], 'Creative': lrcn['personality_creative'], 'Imaginative': lrcn['personality_imaginative'],
            'Ingenious': lrcn['personality_ingenious'], 'Intelligent': lrcn['personality_intelligent'], 'Intuitive': lrcn['personality_intuitive'],
            'Original': lrcn['personality_original'], 'Poetic': lrcn['personality_poetic'], 'Rebellious': lrcn['personality_rebellious'],
            'Obtuse': lrcn['personality_obtuse'], 'Superficial': lrcn['personality_superficial']
        }

        fassess_lrcn = {
            'Arousal': lrcn['final_assess_arousal'],
            'Valence': lrcn['final_assess_valence'], 'Dominance': lrcn['final_assess_dominance'], 'Liking': lrcn['final_assess_liking'],
            'Familiarity': lrcn['final_assess_familiarity'], 'Neutral': lrcn['final_assess_neutral'], 'Disgust': lrcn['final_assess_disgust'],
            'Happiness': lrcn['final_assess_happiness'], 'Surprise': lrcn['final_assess_surprise'], 'Anger': lrcn['final_assess_anger'],
            'Fear': lrcn['final_assess_fear'], 'Sadness': lrcn['final_assess_sadness']
        }

    if not predictions['pred_conv2d'] == {}:
        conv2d = predictions['pred_conv2d']

        panas_conv2d = {
            'Interested': conv2d['panas_interested'], 'Excited': conv2d['panas_excited'], 'Strong': conv2d['panas_strong'],
            'Distressed': conv2d['panas_distressed'], 'Upset': conv2d['panas_upset'], 'Guilty': conv2d['panas_guilty'],
            'Enthusiastic': conv2d['panas_enthusiastic'], 'Proud': conv2d['panas_proud'], 'Alert': conv2d['panas_alert'],
            'Inspired': conv2d['panas_inspired'], 'Scared': conv2d['panas_scared'], 'Hostile': conv2d['panas_hostile'],
            'Irritable': conv2d['panas_irritable'], 'Ashamed': conv2d['panas_ashamed'], 'Determined': conv2d['panas_determined'],
            'Attentive': conv2d['panas_attentive'], 'Active': conv2d['panas_active'], 'Nervous': conv2d['panas_nervous'],
            'Jittery': conv2d['panas_jittery'], 'Afraid': conv2d['panas_afraid']
        }

        personality_conv2d = {
            'Open': conv2d['personality_open'],
            'Warmhearted': conv2d['personality_warmhearted'], 'Extroverted': conv2d['personality_extroverted'],
            'Exuberant': conv2d['personality_exuberant'], 'Vivacious': conv2d['personality_vivacious'], 'Inward_looking': conv2d['personality_inward_looking'],
            'Introverted': conv2d['personality_introverted'], 'Reserved': conv2d['personality_reserved'], 'Silent': conv2d['personality_silent'],
            'Shy': conv2d['personality_shy'], 'Altruistic': conv2d['personality_altruistic'], 'Sympathetic': conv2d['personality_sympathetic'],
            'Agreeable': conv2d['personality_agreeable'], 'Generous': conv2d['personality_generous'], 'Hospitable': conv2d['personality_hospitable'],
            'Cynical': conv2d['personality_cynical'], 'Egocentric': conv2d['personality_egocentric'], 'Egoistic': conv2d['personality_egoistic'],
            'Suspicious': conv2d['personality_suspicious'], 'Revengeful': conv2d['personality_revengeful'], 'Conscientious': conv2d['personality_conscientious'],
            'Diligent': conv2d['personality_diligent'], 'Methodical': conv2d['personality_methodical'], 'Orderly': conv2d['personality_orderly'],
            'Precise': conv2d['personality_precise'], 'Untidy': conv2d['personality_untidy'], 'Careless': conv2d['personality_careless'],
            'Rash': conv2d['personality_rash'], 'Inconstant': conv2d['personality_inconstant'], 'Heedless': conv2d['personality_heedless'],
            'Calm': conv2d['personality_calm'], 'Impassive': conv2d['personality_impassive'], 'Serene': conv2d['personality_serene'],
            'Self_assured': conv2d['personality_self_assured'], 'Anxious': conv2d['personality_anxious'], 'Emotional': conv2d['personality_emotional'],
            'Jealous': conv2d['personality_jealous'], 'Nervous_P': conv2d['personality_nervous_P'], 'Touchy': conv2d['personality_touchy'],
            'Susceptible': conv2d['personality_susceptible'], 'Creative': conv2d['personality_creative'], 'Imaginative': conv2d['personality_imaginative'],
            'Ingenious': conv2d['personality_ingenious'], 'Intelligent': conv2d['personality_intelligent'], 'Intuitive': conv2d['personality_intuitive'],
            'Original': conv2d['personality_original'], 'Poetic': conv2d['personality_poetic'], 'Rebellious': conv2d['personality_rebellious'],
            'Obtuse': conv2d['personality_obtuse'], 'Superficial': conv2d['personality_superficial']
        }

        fassess_conv2d = {
            'Arousal': conv2d['final_assess_arousal'],
            'Valence': conv2d['final_assess_valence'], 'Dominance': conv2d['final_assess_dominance'], 'Liking': conv2d['final_assess_liking'],
            'Familiarity': conv2d['final_assess_familiarity'], 'Neutral': conv2d['final_assess_neutral'], 'Disgust': conv2d['final_assess_disgust'],
            'Happiness': conv2d['final_assess_happiness'], 'Surprise': conv2d['final_assess_surprise'], 'Anger': conv2d['final_assess_anger'],
            'Fear': conv2d['final_assess_fear'], 'Sadness': conv2d['final_assess_sadness']
        }

    filename = filename.replace('\\', '/').split('./')[1]
    instance_path = os.getcwd().replace('\\', '/')
    filename = instance_path + '/' + filename
    os.remove(filename)

    return render_template('predict.html', username=app.config['USERNAME'], also_lrcn=also_lrcn,\
        panas_conv2d=panas_conv2d, personality_conv2d=personality_conv2d, fassess_conv2d=fassess_conv2d,\
        panas_lrcn=panas_lrcn, personality_lrcn=personality_lrcn, fassess_lrcn=fassess_lrcn)

