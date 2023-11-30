import os

from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from googletrans import Translator
import pandas as pd

app = Flask(__name__)

# Load models and data
model_tourism = load_model('model_tourism.h5')
model_hieroglyphs = load_model('model_hieroglyphs.h5')

data_dict = pd.read_excel('data_set.xlsx').set_index('exhibits')['Text in English'].to_dict()

class_tourism_names = ['10_The_HolyQuran', '11_King_Thutmose_III', '12_King_Fouad_I', '13_theVizier_Paser',
                       '14_Sphinxof_theking_Amenemhat_III', '15_Amun_Ra_Kingof_theGods', '16_Nazlet_Khater_Skeleton',
                       '17_Pen_Menkh_TheGovernerOf_Dendara', '18_TheCoffinOf_Lady_Isis', '19_CoffinOf_Nedjemankh',
                       '1_the_female_peasent', '20_TheCoffinOf_Sennedjem', '21_A_silo', '22_Captives_statuettes',
                       '23_Chair_from_the_tomb_of_Queen_Hetepheres', '24_Maat', '25_Mahalawi_water_ewers',
                       '26_Mamluk_Lamps',
                       '27_Khedive_Ismail', '28_Mohamed_Talaat_Pasha_Harb', '29_Model_of_building',
                       '2_statue_ofthe_sphinx',
                       '30_Muhammad_Ali_Pasha', '31_Puplit _of_the_Mosque_of_Abu_Bakr_bin_Mazhar',
                       '32_The_Preist_Psamtik_seneb', '33_The_Madrasaa_and_Mosque_of_Sultan_Hassan',
                       '34_Wekalet_al-Ghouri',
                       '35_The_birth_of_Isis', '36_King_Akhenaten', '37_The_Kiswa_Covering_of_holy_Kaaba',
                       '38_AQueen_in_the_form_of_the_Sphinx', '39_Purification_with_water', '3_Hassan_Fathi',
                       '40_Mashrabiya', '41_Astrolabe', '42_Baker', '43_The_Protective_Godesses', '44_Miller',
                       '45_Hapi_The_Scribe', '46_Thoth', '47_Ottoman_Period_Carpet', '48_Stela_of_King_Qaa',
                       '49_Zainab_Khatun_house', '4_Royal_Statues', '50_God_Nilus', '5_Greek_Statues', '6_Khonsu',
                       '7_Ra_Horakhty', '8_Senenmut', '9_Box_ofthe_Holy Quran', 'Akhenaten', 'Bent pyramid for senefru',
                       'Colossal Statue of Ramesses II', 'Colossoi of Memnon', 'Goddess Isis with her child',
                       'Hatshepsut',
                       'Hatshepsut face', 'Khafre Pyramid', 'Mask of Tutankhamun', 'Nefertiti', 'Pyramid_of_Djoser',
                       'Ramessum', 'Ramses II Red Granite Statue', 'Statue of King Zoser',
                       'Statue of Tutankhamun with Ankhesenamun', 'Temple_of_Isis_in_Philae', 'Temple_of_Kom_Ombo',
                       'The Great Temple of Ramesses II', 'amenhotep iii and tiye', 'bust of ramesses ii',
                       'menkaure pyramid', 'sphinx']

class_hieroglyphs_names = ['100', 'Among', 'Angry', 'Ankh', 'Aroura', 'At', 'Bad_Thinking', 'Bandage', 'Bee', 'Belongs',
                           'Birth', 'Board_Game', 'Book', 'Boy', 'Branch', 'Bread', 'Brewer', 'Builder', 'Bury',
                           'Canal',
                           'Cloth_on_Pole', 'Cobra', 'Composite_Bow', 'Cooked', 'Corpse', 'Dessert', 'Divide', 'Duck',
                           'Elephant', 'Enclosed_Mound', 'Eye', 'Fabric', 'Face', 'Falcon', 'Fingre', 'Fish', 'Flail',
                           'Folded_Cloth', 'Foot', 'Galena', 'Giraffe', 'He', 'Her', 'Hit', 'Horn', 'King', 'Leg',
                           'Length_Of_a_Human_Arm', 'Life_Spirit', 'Limit', 'Lion', 'Lizard', 'Loaf', 'Loaf_Of_Bread',
                           'Man',
                           'Mascot', 'Meet', 'Mother', 'Mouth', 'Musical_Instrument', 'Nile_Fish', 'Not', 'Now',
                           'Nurse',
                           'Nursing', 'Occur', 'One', 'Owl', 'Pair', 'Papyrus_Scroll', 'Pool', 'QuailChick', 'Reed',
                           'Ring',
                           'Rope', 'Ruler', 'Sail', 'Sandal', 'Semen', 'Small_Ring', 'Snake', 'Soldier', 'Star',
                           'Stick',
                           'Swallow', 'This', 'To_Be_Dead', 'To_Protect', 'To_Say', 'Turtle', 'Viper', 'Wall', 'Water',
                           'Woman',
                           'You']


def preprocess_image(image_path):
    image = Image.open(image_path).resize((224, 224))
    return np.array(image) / 255.0


def get_translation(text, lang):
    if lang == 'en':
        return text
    translator = Translator()
    return translator.translate(text, src='en', dest=lang).text


@app.route('/')
def index():
    return render_template('index.html', prediction=None)


@app.route('/predictTourism', methods=['POST'])
def predict_tourism():
    try:
        image_file = request.files['image']
        selected_language = request.form['language']
        image_path = "temp_image.jpg"
        image_file.save(image_path)

        processed_image = preprocess_image(image_path)
        predicted_class = np.argmax(model_tourism.predict(np.expand_dims(processed_image, axis=0)))
        predicted_class_name = class_tourism_names[predicted_class]
        translated_text = get_translation(data_dict.get(predicted_class_name, ''), selected_language)

        return render_template('index.html', prediction=translated_text)

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/predictTourismAPI', methods=['POST'])
def predict_tourism_api():
    try:
        image_file = request.files['image']
        selected_language = request.form['language']
        image_path = "temp_image.jpg"
        image_file.save(image_path)

        processed_image = preprocess_image(image_path)
        predicted_class = np.argmax(model_tourism.predict(np.expand_dims(processed_image, axis=0)))
        predicted_class_name = class_tourism_names[predicted_class]
        translated_text = get_translation(data_dict.get(predicted_class_name, ''), selected_language)

        return jsonify({
            "information": translated_text,
            "name": predicted_class_name
        })

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/predictHieroglyphs', methods=['POST'])
def predict_hieroglyphs():
    try:
        image_file = request.files['image']
        image_path = "temp_image.jpg"
        image_file.save(image_path)

        processed_image = preprocess_image(image_path)
        predicted_class = np.argmax(model_hieroglyphs.predict(np.expand_dims(processed_image, axis=0)))
        predicted_class_name = class_hieroglyphs_names[predicted_class]

        return render_template('index.html', prediction=predicted_class_name)

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/predictHieroglyphsAPI', methods=['POST'])
def predict_hieroglyphs_api():
    try:
        image_file = request.files['image']
        image_path = "temp_image.jpg"
        image_file.save(image_path)

        processed_image = preprocess_image(image_path)
        predicted_class = np.argmax(model_hieroglyphs.predict(np.expand_dims(processed_image, axis=0)))
        predicted_class_name = class_hieroglyphs_names[predicted_class]

        return jsonify({"class": predicted_class_name})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=False, port=int(os.environ.get('PORT', 8080)))

