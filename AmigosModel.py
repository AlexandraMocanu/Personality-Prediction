from keras.models import load_model
from keras.utils import to_categorical
from keras.backend import clear_session
from keras import backend as K
from tensorflow import Session, Graph
from darkflow.net.build import TFNet
from imutils.video import FileVideoStream
import numpy as np
import cv2
import json
import os

class AmigosModel(object):

    def __init__(self):

        #clear keras session
        clear_session()

        self.LRCN_FF = "models/lrcn_frontalface.h5"
        self.LRCN_RGB = "models/lrcn_rgb.h5"
        self.CONV2D_FF = "models/conv2d_frontalface.h5"
        self.CONV2D_RGB = "models/conv2d_rgb.h5"

        self.setup_models_session()

        self.records = "none"
        self.video = "none"
        self.image = "none"

        # if self.arch_type == 'lrcn':
        #     #for now just use rgb
        #     self.rgb_lrcn_model = load_model(self.LRCN_RGB)
        #     self.ff_lrcn_model = load_model(self.LRCN_FF)
        # elif self.arch_type == 'conv2d':
        #     self.rgb_conv2d_model = load_model(self.CONV2D_RGB)
        #     self.ff_conv2d_model = load_model(self.CONV2D_FF)
        # else:

        # load all models at startup
        # self.rgb_lrcn_model = load_model(self.LRCN_RGB)
        # self.rgb_lrcn_model._make_predict_function()
        # self.ff_lrcn_model = load_model(self.LRCN_FF)
        # self.ff_lrcn_model._make_predict_function()
        # self.rgb_conv2d_model = load_model(self.CONV2D_RGB)
        # self.rgb_conv2d_model._make_predict_function()
        # self.ff_conv2d_model = load_model(self.CONV2D_FF)
        # self.ff_conv2d_model._make_predict_function()

        # init yolo
        self.options = {"model": "cfg/tiny-yolo-voc.cfg", "load": "bin/tiny-yolo-voc.weights", "gpu": 1.0}
        self.tfnet = TFNet(self.options)
    

    def set_username(self, username):
        self.username = username
    

    def setup_models_session(self):
        #self.LRCN_FF
        self.graph_lrcn_ff = Graph()
        with self.graph_lrcn_ff.as_default():
            self.session_lrcn_ff = Session(graph=self.graph_lrcn_ff)
            with self.session_lrcn_ff.as_default():
                self.model_lrcn_ff = load_model(self.LRCN_FF)
        
        #self.LRCN_RGB
        self.graph_lrcn_rgb = Graph()
        with self.graph_lrcn_rgb.as_default():
            self.session_lrcn_rgb = Session(graph=self.graph_lrcn_rgb)
            with self.session_lrcn_rgb.as_default():
                self.model_lrcn_rgb = load_model(self.LRCN_RGB)
        
        #self.CONV2D_FF
        self.graph_conv2d_ff = Graph()
        with self.graph_conv2d_ff.as_default():
            self.session_conv2d_ff = Session(graph=self.graph_conv2d_ff)
            with self.session_conv2d_ff.as_default():
                self.model_conv2d_ff = load_model(self.CONV2D_FF)
        
        #self.CONV2D_RGB
        self.graph_conv2d_rgb = Graph()
        with self.graph_conv2d_rgb.as_default():
            self.session_conv2d_rgb = Session(graph=self.graph_conv2d_rgb)
            with self.session_conv2d_rgb.as_default():
                self.model_conv2d_rgb = load_model(self.CONV2D_RGB)


    # arch_type -> one of 'lrcn' or 'conv2d'; choose whether to use the lrcn or conv2d model
    ## lrcn - for video, receive 3 frames
    ##      - for image, copy the image so you have 3 images
    ## conv2d - for video take one frame/second from video and average the results
    ##        - for image, use only that image
    ## default = both
    # model_type -> frontal face or rgb? or maybe just use both and average the results
    #            -> default = both
    # input_type -> one of 'image', 'video'
    def set_config(self, input_type, arch_type='both', model_type='both'):
        self.input_type = input_type
        self.arch_type = arch_type
        self.model_type = model_type
    

    def set_prediction_data(self, records='none', video='none', image='none'):
        self.records = records
        self.video = video
        self.image = image

    
    def load_initial_data(self, initial_data):
        if not initial_data == []:
            init_data = initial_data[0]
            init_data.replace('\\', '/')
            with open(init_data) as json_file:  
                data = json.load(json_file)
        else:
            data = 0
        
        if not data == 0:
            self.input_data = \
                {
                    'age': data['age'],
                    'gender': data['gender'],

                    'Arousal_Initial': data['arousal_in'],
                    'Valence_Initial': data['valence_in'],
                    'Dominance_Initial': data['dominance_in'],
                    'Liking_Initial': data['liking_in'],
                    'Familiarity_Initial': data['familiarity_in'],
                    'Neutral_Initial': data['neutral'],
                    'Disgust_Initial': data['disgust'],
                    'Happiness_Initial': data['happiness'],
                    'Surprise_Initial': data['surprise'],
                    'Anger_Initial': data['anger'],
                    'Fear_Initial': data['fear'],
                    'Sadness_Initial':data['sadness']
                }
        else: # set dummy data in case user doesn't input anything
            self.input_data = \
                {
                    'age': 22,
                    'gender': 0,

                    'Arousal_Initial': 0,
                    'Valence_Initial': 0,
                    'Dominance_Initial': 0,
                    'Liking_Initial': 0,
                    'Familiarity_Initial': 0,
                    'Neutral_Initial': 0,
                    'Disgust_Initial': 0,
                    'Happiness_Initial': 0,
                    'Surprise_Initial': 0,
                    'Anger_Initial': 0,
                    'Fear_Initial': 0,
                    'Sadness_Initial': 0
                }


    def get_frames_imutils(self):
        frames_folder = './records_frames/'+self.username+'/frames'
        if not os.path.exists(frames_folder):
            os.makedirs(frames_folder)

        print("\nExtracting frames ... ")
        
        if not self.video == 'none':
            print("\nProcessing video ...")
            video = self.video
        elif not self.records == 'none':
            print("\nProcessing recording ...")
            video = self.records
        
        frames_path = []
        frames = []
        frames_nb = 0
        
        video = video.replace('\\', '/').split('./')[1]
        instance_path = os.getcwd().replace('\\', '/')
        video = instance_path + '/' + video
        print("\nVIDEO_PATH ... ", video, "\n")
        
        fvs = FileVideoStream(video).start()

        # loop over frames from the video file stream
        while fvs.more():
            frame = fvs.read()
            frame = np.dstack([frame])
            # print("\nSHAPE ... ", np.shape(frame))

            if np.shape(frame) != (1, 1, 1):
                # person prediction
                result = self.tfnet.return_predict(frame)

                for i in result:
                    if i['label'] == 'person':
                        topleft_x = i['topleft']['x'] #x1
                        topleft_y = i['topleft']['y'] #y1
                        bottomright_x = i['bottomright']['x'] #x2
                        bottomright_y = i['bottomright']['y'] #y2
                        
                        croped_img = frame[topleft_y:bottomright_y, topleft_x:bottomright_x].copy()
                        croped_img = cv2.resize(croped_img, (480, 512))
                        
                        path_f = os.path.join(frames_folder, self.username+'_frame_'+str(frames_nb)+'.jpg')
                        cv2.imwrite(path_f, croped_img)
                        frames_nb += 1
                        frames_path.append(path_f)
                        frames.append(croped_img)

                        break # to make sure that no more than 1 person is found!
        fvs.stop()
        
        print("\nFRAMES SIZE ... ", len(frames))
        return frames


    # get and save frames from video
    def get_frames(self):
        frames_folder = './records_frames/'+self.username+'/frames'
        if not os.path.exists(frames_folder):
            os.makedirs(frames_folder)

        print("\nExtracting frames ... ")
        
        if not self.video == 'none':
            print("\nProcessing video ...")
            video = self.video
        elif not self.records == 'none':
            print("\nProcessing recording ...")
            video = self.records
        
        frames_path = []
        frames = []
        frames_nb = 0
        
        video = video.replace('\\', '/').split('./')[1]
        instance_path = os.getcwd().replace('\\', '/')
        video = instance_path + '/' + video
        print("\nVIDEO_PATH ... ", video, "\n")
        vcap = cv2.VideoCapture(video)
        framecount = 0
        if vcap.isOpened():
            framerate_video = vcap.get(cv2.CAP_PROP_FPS)
            print("\nFPS ... ", framerate_video)
            # if framerate_video > 20:
            #     framerate = 200 #process one every 200 frames
            # elif framerate_video > 10:
            #     framerate = 100 #process one every 100 frames
            # else:
            framerate = 10

            while True:
                ret, frame = vcap.read()

                if ret == True:
                    # if framecount%framerate == 1:
                    result = self.tfnet.return_predict(frame)

                    for i in result:
                        if i['label'] == 'person':
                            topleft_x = i['topleft']['x'] #x1
                            topleft_y = i['topleft']['y'] #y1
                            bottomright_x = i['bottomright']['x'] #x2
                            bottomright_y = i['bottomright']['y'] #y2
                            
                            croped_img = frame[topleft_y:bottomright_y, topleft_x:bottomright_x].copy()
                            croped_img = cv2.resize(croped_img, (480, 512))
                            
                            path_f = os.path.join(frames_folder, self.username+'_frame_'+str(frames_nb)+'.jpg')
                            cv2.imwrite(path_f, croped_img)
                            frames_nb += 1
                            frames_path.append(path_f)
                            frames.append(croped_img)

                            break # to make sure that no more than 1 person is found!
                    
                    # framecount += 1
                else:
                    print("\nRet false ...")
                    break
        vcap.release()

        print("\nFRAMES SIZE ... ", len(frames))
        return frames
    

    def get_image(self):
        image = cv2.imread(self.image)
        image = cv2.resize(image, (480, 512))
        return image


    def create_prediction_data(self, frames='none', image='none'):
        inputs_lrcn = []
        inputs_conv2d = []

        if frames != 'none':
            if self.arch_type == 'lrcn' or self.arch_type == 'both': # only 3 frames
                first = frames[0]
                middle = frames[(int(len(frames)/2) - 1)]
                end = frames[len(frames)-1]

                images = np.stack((first, middle, end), axis=0)

                inputs_f = {
                    'images' : np.expand_dims(images, axis=0), # in the future, with the new model, it actually should be image_bytes!
                    'age': np.expand_dims(self.input_data['age'], axis=0),
                    'gender': np.expand_dims(to_categorical(self.input_data['gender'], 2), axis=0),
                    'Arousal_Initial': np.expand_dims(to_categorical(self.input_data['Arousal_Initial'], 10), axis=0),
                    'Valence_Initial': np.expand_dims(to_categorical(self.input_data['Valence_Initial'], 10), axis=0),
                    'Dominance_Initial': np.expand_dims(to_categorical(self.input_data['Dominance_Initial'], 10), axis=0),
                    'Liking_Initial': np.expand_dims(to_categorical(self.input_data['Liking_Initial'], 10), axis=0),
                    'Familiarity_Initial': np.expand_dims(to_categorical(self.input_data['Familiarity_Initial'], 10), axis=0),
                    'Neutral_Initial': np.expand_dims(to_categorical(self.input_data['Neutral_Initial'], 2), axis=0),
                    'Disgust_Initial': np.expand_dims(to_categorical(self.input_data['Disgust_Initial'], 2), axis=0),
                    'Happiness_Initial': np.expand_dims(to_categorical(self.input_data['Happiness_Initial'], 2), axis=0),
                    'Surprise_Initial': np.expand_dims(to_categorical(self.input_data['Surprise_Initial'], 2), axis=0),
                    'Anger_Initial': np.expand_dims(to_categorical(self.input_data['Anger_Initial'], 2), axis=0),
                    'Fear_Initial': np.expand_dims(to_categorical(self.input_data['Fear_Initial'], 2), axis=0),
                    'Sadness_Initial': np.expand_dims(to_categorical(self.input_data['Sadness_Initial'], 2), axis=0)
                }

                inputs_lrcn.append(inputs_f)

            if self.arch_type == 'conv2d' or self.arch_type == 'both': # all frames
                for frame in frames:
                    inputs_f = {
                        'image' : np.expand_dims(np.array(frame), axis=0), # in the future, with the new model, it actually should be image_bytes!
                        'age': np.expand_dims(self.input_data['age'], axis=0),
                        'gender': np.expand_dims(to_categorical(self.input_data['gender'], 2), axis=0),
                        'Arousal_Initial': np.expand_dims(to_categorical(self.input_data['Arousal_Initial'], 10), axis=0),
                        'Valence_Initial': np.expand_dims(to_categorical(self.input_data['Valence_Initial'], 10), axis=0),
                        'Dominance_Initial': np.expand_dims(to_categorical(self.input_data['Dominance_Initial'], 10), axis=0),
                        'Liking_Initial': np.expand_dims(to_categorical(self.input_data['Liking_Initial'], 10), axis=0),
                        'Familiarity_Initial': np.expand_dims(to_categorical(self.input_data['Familiarity_Initial'], 10), axis=0),
                        'Neutral_Initial': np.expand_dims(to_categorical(self.input_data['Neutral_Initial'], 2), axis=0),
                        'Disgust_Initial': np.expand_dims(to_categorical(self.input_data['Disgust_Initial'], 2), axis=0),
                        'Happiness_Initial': np.expand_dims(to_categorical(self.input_data['Happiness_Initial'], 2), axis=0),
                        'Surprise_Initial': np.expand_dims(to_categorical(self.input_data['Surprise_Initial'], 2), axis=0),
                        'Anger_Initial': np.expand_dims(to_categorical(self.input_data['Anger_Initial'], 2), axis=0),
                        'Fear_Initial': np.expand_dims(to_categorical(self.input_data['Fear_Initial'], 2), axis=0),
                        'Sadness_Initial': np.expand_dims(to_categorical(self.input_data['Sadness_Initial'], 2), axis=0)
                    }
                    
                    inputs_conv2d.append(inputs_f)
            
        elif not image == 'none':
            inputs_f = {
                'image' : np.expand_dims(np.array(image), axis=0), # in the future, with the new model, it actually should be image_bytes!
                'age': np.expand_dims(self.input_data['age'], axis=0),
                'gender': np.expand_dims(to_categorical(self.input_data['gender'], 2), axis=0),
                'Arousal_Initial': np.expand_dims(to_categorical(self.input_data['Arousal_Initial'], 10), axis=0),
                'Valence_Initial': np.expand_dims(to_categorical(self.input_data['Valence_Initial'], 10), axis=0),
                'Dominance_Initial': np.expand_dims(to_categorical(self.input_data['Dominance_Initial'], 10), axis=0),
                'Liking_Initial': np.expand_dims(to_categorical(self.input_data['Liking_Initial'], 10), axis=0),
                'Familiarity_Initial': np.expand_dims(to_categorical(self.input_data['Familiarity_Initial'], 10), axis=0),
                'Neutral_Initial': np.expand_dims(to_categorical(self.input_data['Neutral_Initial'], 2), axis=0),
                'Disgust_Initial': np.expand_dims(to_categorical(self.input_data['Disgust_Initial'], 2), axis=0),
                'Happiness_Initial': np.expand_dims(to_categorical(self.input_data['Happiness_Initial'], 2), axis=0),
                'Surprise_Initial': np.expand_dims(to_categorical(self.input_data['Surprise_Initial'], 2), axis=0),
                'Anger_Initial': np.expand_dims(to_categorical(self.input_data['Anger_Initial'], 2), axis=0),
                'Fear_Initial': np.expand_dims(to_categorical(self.input_data['Fear_Initial'], 2), axis=0),
                'Sadness_Initial': np.expand_dims(to_categorical(self.input_data['Sadness_Initial'], 2), axis=0)
            }

            inputs_conv2d.append(inputs_f)
        
        if inputs_lrcn == []:
            return {
                'lrcn': 0,
                'conv2d': np.array(inputs_conv2d)
            }
        else:
            return {
                'lrcn': np.array(inputs_lrcn),
                'conv2d': np.array(inputs_conv2d)
            }


    def predict_video(self, prediction_data, image='No'):
        predictions_labels = ['panas_interested', 'panas_excited', 'panas_strong', 'panas_distressed', 'panas_upset',
                'panas_guilty', 'panas_enthusiastic', 'panas_proud', 'panas_alert', 'panas_inspired',
                'panas_scared', 'panas_hostile', 'panas_irritable', 'panas_ashamed',
                'panas_determined', 'panas_attentive', 'panas_active', 'panas_nervous', 'panas_jittery', 'panas_afraid',
                'personality_open', 'personality_warmhearted', 'personality_extroverted', 'personality_exuberant',
                'personality_vivacious', 'personality_inward_looking', 'personality_introverted', 'personality_reserved',
                'personality_silent', 'personality_shy', 'personality_altruistic', 'personality_sympathetic',
                'personality_agreeable', 'personality_generous', 'personality_hospitable', 'personality_cynical',
                'personality_egocentric', 'personality_egoistic', 'personality_suspicious', 'personality_revengeful',
                'personality_conscientious', 'personality_diligent', 'personality_methodical', 'personality_orderly',
                'personality_precise', 'personality_untidy', 'personality_careless', 'personality_rash',
                'personality_inconstant', 'personality_heedless', 'personality_calm', 'personality_impassive',
                'personality_serene', 'personality_self_assured', 'personality_anxious', 'personality_emotional',
                'personality_jealous', 'personality_nervous_P', 'personality_touchy', 'personality_susceptible',
                'personality_creative', 'personality_imaginative', 'personality_ingenious', 'personality_intelligent',
                'personality_intuitive', 'personality_original', 'personality_poetic', 'personality_rebellious',
                'personality_obtuse', 'personality_superficial',
                'final_assess_arousal', 'final_assess_valence', 'final_assess_dominance', 'final_assess_liking',
                'final_assess_familiarity', 'final_assess_neutral', 'final_assess_disgust', 'final_assess_happiness',
                'final_assess_surprise', 'final_assess_anger', 'final_assess_fear', 'final_assess_sadness']

        print("\nSAMPLES NB ... ", len(prediction_data['conv2d'][0]))

        # TODO: verbose=1
        if image == 'No': #we have video
            # if self.rgb_lrcn_model:
                #clear keras session
            clear_session()
            print("\n Loading and calling LRCN - RGB Model ... ")
            # self.rgb_lrcn_model = load_model(self.LRCN_RGB)
            # self.rgb_lrcn_model._make_predict_function()
            K.set_session(self.session_lrcn_rgb)
            with self.graph_lrcn_rgb.as_default():
                results_pred_rgb_lrcn = self.model_lrcn_rgb.predict(prediction_data['lrcn'][0], batch_size=1)
            
            # if self.ff_lrcn_model:
                #clear keras session
            clear_session()
            print("\n Loading and calling LRCN - FF Model ... ")
            # self.ff_lrcn_model = load_model(self.LRCN_FF)
            # self.ff_lrcn_model._make_predict_function()
            K.set_session(self.session_lrcn_ff)
            with self.graph_lrcn_ff.as_default():
                results_pred_ff_lrcn = self.model_lrcn_ff.predict(prediction_data['lrcn'][0], batch_size=1)
                        
            # if self.rgb_conv2d_model:
                #clear keras session
            clear_session()
            print("\n Loading and calling Conv2D - RGB Model ... ")
            # self.rgb_conv2d_model = load_model(self.CONV2D_RGB)
            # self.rgb_conv2d_model._make_predict_function()
            K.set_session(self.session_conv2d_rgb)
            with self.graph_conv2d_rgb.as_default():
                results_pred_rgb_conv2d = self.model_conv2d_rgb.predict(prediction_data['conv2d'][0], batch_size=len(prediction_data['conv2d'][0]))
            
            # if self.ff_conv2d_model:
                #clear keras session
            clear_session()
            print("\n Loading and calling Conv2D - FF Model ... ")
            # self.ff_conv2d_model = load_model(self.CONV2D_FF)
            # self.ff_conv2d_model._make_predict_function()
            K.set_session(self.session_conv2d_ff)
            with self.graph_conv2d_ff.as_default():
                results_pred_ff_conv2d = self.model_conv2d_ff.predict(prediction_data['conv2d'][0], batch_size=len(prediction_data['conv2d'][0]))
        else:
            # if self.rgb_conv2d_model:
                #clear keras session
            clear_session()
            print("\n Loading and calling Conv2D - RGB Model ... ")
            # self.rgb_conv2d_model = load_model(self.CONV2D_RGB)
            # self.rgb_conv2d_model._make_predict_function()
            K.set_session(self.session_conv2d_rgb)
            with self.graph_conv2d_rgb.as_default():
                results_pred_rgb_conv2d = self.model_conv2d_rgb.predict(prediction_data['conv2d'][0], batch_size=1)
            # if self.ff_conv2d_model:
                #clear keras session
            clear_session()
            print("\n Loading and calling Conv2D - FF Model ... ")
            # self.ff_conv2d_model = load_model(self.CONV2D_FF)
            # self.ff_conv2d_model._make_predict_function()
            K.set_session(self.session_conv2d_ff)
            with self.graph_conv2d_ff.as_default():
                results_pred_ff_conv2d = self.model_conv2d_ff.predict(prediction_data['conv2d'][0], batch_size=1)

        # Empty dict of predictions
        predictions_amigos_lrcn = {}
        predictions_amigos_conv2d = {}
        
        for pred in range(len(predictions_labels)):
            label = predictions_labels[pred]

            if image == 'No' and not self.arch_type == 'conv2d':
                result_rgb_lrcn = np.argmax(results_pred_rgb_lrcn[pred][0])
                result_ff_lrcn = np.argmax(results_pred_ff_lrcn[pred][0])

                result_lrcn = int((int(result_rgb_lrcn) + int(result_ff_lrcn))/2)

                predictions_amigos_lrcn[label] = result_lrcn
            
            if not self.arch_type == 'lrcn':
                result_rgb_conv2d = np.argmax(results_pred_rgb_conv2d[pred][0])
                result_ff_conv2d = np.argmax(results_pred_ff_conv2d[pred][0])

                result_conv2d = int((int(result_rgb_conv2d) + int(result_ff_conv2d))/2)

                predictions_amigos_conv2d[label] = result_conv2d

        return {
            'pred_lrcn': predictions_amigos_lrcn,
            'pred_conv2d': predictions_amigos_conv2d
        }
    

    def get_predictions(self):

        #clear keras session
        clear_session()

        # get image or video-frames
        if not self.records == 'none':
            frames = self.get_frames_imutils()
            prediction_data = self.create_prediction_data(frames=frames)
            self.predictions = self.predict_video(prediction_data)
        elif not self.video == 'none':
            frames = self.get_frames()
            prediction_data = self.create_prediction_data(frames=frames)
            self.predictions = self.predict_video(prediction_data)
        elif not self.image == 'none':
            image = self.get_image()
            prediction_data = self.create_prediction_data(image=image)
            self.predictions = self.predict_video(prediction_data, image='yes')
        
        return self.predictions