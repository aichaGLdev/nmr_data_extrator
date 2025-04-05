from flask import Flask, request, jsonify, send_file # Importation des modules Flask nécessaires

from CutImg import scale_detect
from HorizontalBracket import detect_horizontal_bracket_andNum ,detectcntrnum ,detect_wntr_num ,convert_to_actual_bracket
from Mf import index_molecular_formula
from OCR import OCR
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import cv2  # Importation de la bibliothèque OpenCV
import os  # Importation du module os pour la manipulation de fichiers et de répertoires
import time
from flask_cors import CORS

app = Flask(__name__)  # Initialisation de l'application Flask
CORS(app)  # This will enable CORS for all routes

ocr = OCR()
load_dotenv('./.env')

trainsetpath = "./trainset"  # Chemin vers le répertoire contenant les données d'entraînement
testsetpath = "./testset"
ocr.knn_instance.KNNTrain(trainsetpath)
ocr.knn_instance.KNNTestSet(testsetpath)
print("Train & Test KNN terminé !")
port = os.getenv('PORT')
UPLOAD_FOLDER = os.getenv('UPLOAD_DIRECTORY')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'H' not in request.files or 'C' not in request.files:
        return jsonify({'error': 'Both H and C files are required'}), 400

    h1_file = request.files['H']  # Changed to 'H'
    c13_file = request.files['C']  # Changed to 'C'

    if h1_file.filename == '' or c13_file.filename == '':
        return jsonify({'error': 'No selected files'}), 400
    name = request.form.get('name')
    mf = request.form.get('mf')
    id_trail = request.form.get('id_trail')
    if not name or not mf:
        return jsonify({'error': 'Name and mf are required'}), 400

    if h1_file and c13_file:
        UPLOAD_DIRECTORY = os.getenv('UPLOAD_DIRECTORY', 'uploads')
        h1_filepath = os.path.join(app.config['UPLOAD_FOLDER'],  f"{UPLOAD_DIRECTORY}{id_trail}H.png")
        c13_filepath = os.path.join(app.config['UPLOAD_FOLDER'],  f"{UPLOAD_DIRECTORY}{id_trail}C.png")

        h1_file.save(h1_filepath)
        c13_file.save(c13_filepath)

        # Process the files and additional data as needed
        filepath = f"{UPLOAD_DIRECTORY}/uploads{id_trail}C.png"
        Hfilepath = f"{UPLOAD_DIRECTORY}/uploads{id_trail}H.png"
        indexed_formula, carbon_count = index_molecular_formula(mf)
        image = cv2.imread(filepath, cv2.IMREAD_COLOR)
        numbers = []
        bracket_pos = []
        detect_wntr_num(image, ocr, numbers, bracket_pos)
        h_numbers = []
        h_bracket_pos = []

        # Détection de l'échelle et des positions des crochets horizontaux pour l'image 1H
        h_image = cv2.imread(Hfilepath, cv2.IMREAD_COLOR)
        if h_image is None:
            return jsonify({'error': 'Failed to load 1H image'}), 500

        scale = scale_detect(h_image, ocr)
        if not scale:
            return jsonify({'error': 'Failed to detect scale'}), 500

        detect_horizontal_bracket_andNum(h_image, ocr, h_numbers, h_bracket_pos)
        if not h_numbers or not h_bracket_pos:
            return jsonify({'error': 'Failed to detect brackets or numbers in 1H image'}), 500

        actual_bracket = convert_to_actual_bracket(scale, h_bracket_pos)
        if not actual_bracket:
            return jsonify({'error': 'Failed to convert brackets to actual positions'}), 500

        hydrogen_info = []
        for k in range(min(len(actual_bracket), len(h_numbers))):
            shift_avg = (actual_bracket[k][0] + actual_bracket[k][1]) / 2
            hydrogen_info.append({
                "integral": round(float(h_numbers[k])),
                "shift": shift_avg
            })

        # Filtrer les pics avec un déplacement chimique entre 60 et 80 ppm
        filtered_numbers = [nmr_shift for nmr_shift in numbers if not (60 <= float(nmr_shift) <= 80)]
        if carbon_count == len(filtered_numbers):
            print("infos---------------------------------------------------\n")
            print(filepath + '\n')
            carbon_info = []
            correlations = []

            for index, nmr_shift in enumerate(filtered_numbers, start=1):
                info = {
                    "index": index,
                    "symbol": "C"
                }
                if float(nmr_shift) <= 90:
                    if float(nmr_shift) < 30:
                        info["hybridization"] = 3  # sp3
                        info["multiplicity"] = 3
                        corr = {
                            "atom1": index,
                            "atom2": None,
                            "atom2_suggestion": f"the chemical shift value is between 0 and 30 , which means that the carbon {index} must be directly bonded to an sp3 carbon",
                            "correlationType": "BOND"
                        }
                        correlations.append(corr)
                    else:
                        info["hybridization"] = 3  # sp3
                        info["multiplicity"] = -1
                        info["multiplicity_suggestion"] = "The multiplicity can be 1 or 2"
                        corr = {
                            "atom1": index,
                            "atom2": None,
                            "atom2_suggestion": f"the chemical shift value is between 30 and 90 , which means that the carbon {index} can not be directly bonded to a  carbon",
                            "correlationType": "BOND"
                        }
                        correlations.append(corr)
                elif 90 < float(nmr_shift) <= 130:
                    info["hybridization"] = 2  # sp2
                    info["multiplicity"] = 1
                    if 118 < float(nmr_shift):
                        corr1 = {
                            "atom1": index,
                            "atom2": None,
                            "atom2_suggestion": f"the chemical shift value is between 118 and 130 , which means that the carbon {index} must be directly bonded to an sp3 carbon",
                            "correlationType": "BOND"
                        }
                        corr2 = {
                            "atom1": index,
                            "atom2": None,
                            "atom2_suggestion": f"the chemical shift value is between 118 and 130 , which means that the carbon {index} must be directly bonded to an sp2 carbon",
                            "correlationType": "BOND"
                        }
                        correlations.extend([corr1, corr2])
                else:
                    if 155 < float(nmr_shift) <= 180:
                        info["hybridization"] = 2  # sp2
                        info["multiplicity"] = -1
                        #or 2 a reviser
                        info["multiplicity_suggestion"] = "The multiplicity can be 0 or 1 or 2"
                        corr1 = {
                            "atom1": index,
                            "atom2": None,
                            "atom2_suggestion": f"the chemical shift value is between 155 and 180 , which means that the carbon {index} must be directly bonded to an sp2 carbon or an sp2 oxygen ",
                            "correlationType": "BOND"
                        }
                        corr2 = {
                            "atom1": index,
                            "atom2": None,
                            "atom2_suggestion": f"the chemical shift value is between 155 and 180 , which means that the carbon {index} can be directly bonded to an sp3 carbon ",
                            "correlationType": "BOND"
                        }
                        corr3 = {
                            "atom1": index,
                            "atom2": None,
                            "atom2_suggestion": f"the chemical shift value is between 130 and 155 , which means that the carbon {index} can  be directly bonded to a sp3 oxygen",
                            "correlationType": "BOND"
                        }
                        correlations.extend([corr1, corr2, corr3])
                    else:
                        if 130 < float(nmr_shift) <= 155:
                            corr1 = {
                                "atom1": index,
                                "atom2": None,
                                "atom2_suggestion": f"the chemical shift value is between 130 and 155 , which means that the carbon {index} must be directly bonded to an sp3 carbon",
                                "correlationType": "BOND"
                            }
                            corr2 = {
                                "atom1": index,
                                "atom2": None,
                                "atom2_suggestion": f"the chemical shift value is between 130 et 155 , which means that the carbon {index} must be directly bonded to an sp2 carbon",
                                "correlationType": "BOND"
                            }
                            corr3 = {
                                "atom1": index,
                                "atom2": None,
                                "atom2_suggestion": f"the chemical shift value is between 130 et 155 , which means that the carbon {index} can  be directly bonded to a sp3 carbon or an OR group",
                                "correlationType": "BOND"
                            }
                            correlations.extend([corr1, corr2, corr3])
                        elif 180 < float(nmr_shift) :
                            corr1 = {
                                "atom1": index,
                                "atom2": None,
                                "atom2_suggestion": f"the chemical shift value is between 130 and 155 , which means that the carbon {index} can be directly bonded to an sp2 oxygen",
                                "correlationType": "BOND"
                            }
                            corr2 = {
                                "atom1": index,
                                "atom2": None,
                                "atom2_suggestion": f"the chemical shift value is between 130 and 155 , which means that the carbon {index} must be directly bonded to an sp3 carbon",
                                "correlationType": "BOND"
                            }
                            corr3 = {
                                "atom1": index,
                                "atom2": None,
                                "atom2_suggestion": f"the chemical shift value is between 130 and 155 , which means that the carbon {index} can  be directly bonded to an sp3 carbon",
                                "correlationType": "BOND"
                            }
                            correlations.extend([corr1, corr2, corr3])
                            if 200 < float(nmr_shift):
                                corr1 = {
                                    "atom1": index,
                                    "atom2": None,
                                    "atom2_suggestion": f"the chemical shift value is between 130 and 155 , which means that the carbon {index} can be directly bonded to an sp2 carbon or an sp2 Soufre",
                                    "correlationType": "BOND"
                                }
                                correlations.extend([corr1])
                        info["hybridization"] = 2  # sp2
                        info["multiplicity"] = 0
                info["nmr_shift"] = nmr_shift
                carbon_info.append(info)
            result = {
                "name": name,
                "mf": mf,
                "desc": f"{name} organic molecule",
                "atoms": carbon_info,
                "correlations": correlations,
                "hydrogen": hydrogen_info
            }
            response = True
            print(result)
            time1 = time.time()
            print(f"time1 = {time1}\n")
            print("---------------------------------------------------\n\n")
            return jsonify({'success': response, 'result': result})
        else:
            response = False
            return jsonify({'success': response}), 400


@app.route('/ProcessSpectrums', methods=['POST'])
def ProcessSpectrums():
    print("Received form data:", request.form)
    # Récupérer les paramètres du formulaire
    name = request.form.get('name')
    mf = request.form.get('mf')
    id_trail = request.form.get('id_trail')

    if not name or not mf or not id_trail:
        return jsonify({'error': 'Name, mf, and id_trail are required'}), 400

    UPLOAD_DIRECTORY = os.getenv('UPLOAD_DIRECTORY', 'uploads')
    h1_filepath = os.path.join(UPLOAD_DIRECTORY, f"{id_trail}H.png")
    c13_filepath = os.path.join(UPLOAD_DIRECTORY, f"{id_trail}C.png")

    if not os.path.isfile(h1_filepath) or not os.path.isfile(c13_filepath):
        return jsonify({'error': 'Files not found'}), 404

    # Lire les images
    h1_image = cv2.imread(h1_filepath, cv2.IMREAD_COLOR)
    c13_image = cv2.imread(c13_filepath, cv2.IMREAD_COLOR)

    if h1_image is None or c13_image is None:
        return jsonify({'error': 'Failed to load images'}), 500

    # Traitement de l'image 13C
    indexed_formula, carbon_count = index_molecular_formula(mf)
    image = cv2.imread(c13_filepath, cv2.IMREAD_COLOR)
    numbers = []
    bracket_pos = []
    detect_wntr_num(image, ocr, numbers, bracket_pos)
    h_numbers = []
    h_bracket_pos = []

    # Détection de l'échelle et des positions des crochets horizontaux pour l'image 1H
    h_image = cv2.imread(h1_filepath, cv2.IMREAD_COLOR)
    if h_image is None:
        return jsonify({'error': 'Failed to load 1H image'}), 500

    scale = scale_detect(h_image, ocr)
    if not scale:
        return jsonify({'error': 'Failed to detect scale'}), 500

    detect_horizontal_bracket_andNum(h_image, ocr, h_numbers, h_bracket_pos)
    if not h_numbers or not h_bracket_pos:
        return jsonify({'error': 'Failed to detect brackets or numbers in 1H image'}), 500

    actual_bracket = convert_to_actual_bracket(scale, h_bracket_pos)
    if not actual_bracket:
        return jsonify({'error': 'Failed to convert brackets to actual positions'}), 500

    hydrogen_info = []
    for k in range(min(len(actual_bracket), len(h_numbers))):
        shift_avg = (actual_bracket[k][0] + actual_bracket[k][1]) / 2
        hydrogen_info.append({
            "integral": round(float(h_numbers[k])),
            "shift": shift_avg
        })

    # Filtrer les pics avec un déplacement chimique entre 60 et 80 ppm
    filtered_numbers = [nmr_shift for nmr_shift in numbers if not (60 <= float(nmr_shift) <= 80)]
    if carbon_count == len(filtered_numbers):
        print("infos---------------------------------------------------\n")
        print(c13_filepath + '\n')
        carbon_info = []
        correlations = []

        for index, nmr_shift in enumerate(filtered_numbers, start=1):
            info = {
                "index": index,
                "symbol": "C"
            }
            if float(nmr_shift) <= 90:
                if float(nmr_shift) < 30:
                    info["hybridization"] = 3  # sp3
                    info["multiplicity"] = 3
                    corr = {
                        "atom1": index,
                        "atom2": None,
                        "atom2_suggestion": f"the chemical shift value is between 0 and 30 , which means that the carbon {index} must be directly bonded to an sp3 carbon",
                        "correlationType": "BOND"
                    }
                    correlations.append(corr)
                else:
                    info["hybridization"] = 3  # sp3
                    info["multiplicity"] = None
                    info["multiplicity_suggestion"] = "The multiplicity can be 1 or 2"
                    corr = {
                        "atom1": index,
                        "atom2": None,
                        "atom2_suggestion": f"the chemical shift value is between 30 and 90 , which means that the carbon {index} can not be directly bonded to a  carbon",
                        "correlationType": "BOND"
                    }
                    correlations.append(corr)
            elif 90 < float(nmr_shift) <= 130:
                info["hybridization"] = 2  # sp2
                info["multiplicity"] = 1
                if 118 < float(nmr_shift):
                    corr1 = {
                        "atom1": index,
                        "atom2": None,
                        "atom2_suggestion": f"the chemical shift value is between 118 and 130 , which means that the carbon {index} must be directly bonded to an sp3 carbon",
                        "correlationType": "BOND"
                    }
                    corr2 = {
                        "atom1": index,
                        "atom2": None,
                        "atom2_suggestion": f"the chemical shift value is between 118 and 130 , which means that the carbon {index} must be directly bonded to an sp2 carbon",
                        "correlationType": "BOND"
                    }
                    correlations.extend([corr1, corr2])
            else:
                if 155 < float(nmr_shift) <= 180:
                    info["hybridization"] = 2  # sp2
                    info["multiplicity"] = None
                    # or 2 a reviser
                    info["multiplicity_suggestion"] = "The multiplicity can be 0 or 1 or 2"
                    corr1 = {
                        "atom1": index,
                        "atom2": None,
                        "atom2_suggestion": f"the chemical shift value is between 155 and 180 , which means that the carbon {index} must be directly bonded to an sp2 carbon or an sp2 oxygen ",
                        "correlationType": "BOND"
                    }
                    corr2 = {
                        "atom1": index,
                        "atom2": None,
                        "atom2_suggestion": f"the chemical shift value is between 155 and 180 , which means that the carbon {index} can be directly bonded to an sp3 carbon ",
                        "correlationType": "BOND"
                    }
                    corr3 = {
                        "atom1": index,
                        "atom2": None,
                        "atom2_suggestion": f"the chemical shift value is between 130 and 155 , which means that the carbon {index} can  be directly bonded to a sp3 oxygen",
                        "correlationType": "BOND"
                    }
                    correlations.extend([corr1, corr2, corr3])
                else:
                    if 130 < float(nmr_shift) <= 155:
                        corr1 = {
                            "atom1": index,
                            "atom2": None,
                            "atom2_suggestion": f"the chemical shift value is between 130 and 155 , which means that the carbon {index} must be directly bonded to an sp3 carbon",
                            "correlationType": "BOND"
                        }
                        corr2 = {
                            "atom1": index,
                            "atom2": None,
                            "atom2_suggestion": f"the chemical shift value is between 130 et 155 , which means that the carbon {index} must be directly bonded to an sp2 carbon",
                            "correlationType": "BOND"
                        }
                        corr3 = {
                            "atom1": index,
                            "atom2": None,
                            "atom2_suggestion": f"the chemical shift value is between 130 et 155 , which means that the carbon {index} can  be directly bonded to a sp3 carbon or an OR group",
                            "correlationType": "BOND"
                        }
                        correlations.extend([corr1, corr2, corr3])
                    elif 180 < float(nmr_shift):
                        corr1 = {
                            "atom1": index,
                            "atom2": None,
                            "atom2_suggestion": f"the chemical shift value is between 130 and 155 , which means that the carbon {index} can be directly bonded to an sp2 oxygen",
                            "correlationType": "BOND"
                        }
                        corr2 = {
                            "atom1": index,
                            "atom2": None,
                            "atom2_suggestion": f"the chemical shift value is between 130 and 155 , which means that the carbon {index} must be directly bonded to an sp3 carbon",
                            "correlationType": "BOND"
                        }
                        corr3 = {
                            "atom1": index,
                            "atom2": None,
                            "atom2_suggestion": f"the chemical shift value is between 130 and 155 , which means that the carbon {index} can  be directly bonded to an sp3 carbon",
                            "correlationType": "BOND"
                        }
                        correlations.extend([corr1, corr2, corr3])
                        if 200 < float(nmr_shift):
                            corr1 = {
                                "atom1": index,
                                "atom2": None,
                                "atom2_suggestion": f"the chemical shift value is between 130 and 155 , which means that the carbon {index} can be directly bonded to an sp2 carbon or an sp2 Soufre",
                                "correlationType": "BOND"
                            }
                            correlations.extend([corr1])
                    info["hybridization"] = 2  # sp2
                    info["multiplicity"] = 0
            info["nmr_shift"] = nmr_shift
            carbon_info.append(info)
        result = {
            "name": name,
            "mf": mf,
            "desc": f"{name} organic molecule",
            "atoms": carbon_info,
            "correlations": correlations,
            "hydrogen": hydrogen_info
        }
        response = True
        print(result)
        time1 = time.time()
        print(f"time1 = {time1}\n")
        print("---------------------------------------------------\n\n")
        return jsonify({'success': response, 'result': result})
    else:
        response = False
        return jsonify({'success': response}), 400



@app.route('/extract_1H_chemical_shift', methods=['GET'])
def extract_1H_chemical_shift():
    filepath = "./examples-input/12H.png"
    image = cv2.imread(filepath, cv2.IMREAD_COLOR)
    numbers = []
    bracket_pos = []
    print(filepath)

    # Placeholder function calls and outputs (replace with actual implementations)
    # Assuming these functions are already defined somewhere
    scale = scale_detect(image, ocr)  # Placeholder for scale detection function
    detectcntrnum(image, ocr, numbers, bracket_pos)  # Placeholder for number and bracket detection

    print("numbers", numbers)
    actual_bracket = convert_to_actual_bracket(scale, bracket_pos)  # Convert to actual brackets

    brackets_info = []
    for k in range(min(len(actual_bracket), len(numbers))):
        brackets_info.append((numbers[k], actual_bracket[k][0], actual_bracket[k][1]))

    print("brackets_info", brackets_info)

    # Log the brackets info
    for bracket in brackets_info:
        print(f"( {bracket[0]}, {bracket[1]}, {bracket[2]} )")

    time1 = time.time()
    print("time", time1)


    return "extract 1H chemical shift done !"

@app.route('/process_image', methods=['GET'])
def test_function():
    filepath = "./examples-input/1H.png"

    # Lecture de l'image
    image = cv2.imread(filepath, cv2.IMREAD_COLOR)
    numbers = []
    bracket_pos = []

    # Détection de l'échelle et des positions des crochets horizontaux
    scale = scale_detect(image, ocr)
    # Afficher "scale start"
    print("scale start")

    # Afficher les valeurs de l'échelle
    for i, value in enumerate(scale):
        print(f"scale[{i}] = {value}")
    detect_horizontal_bracket_andNum(image, ocr, numbers, bracket_pos)

    print("integral---------------------------------------------------")
    print(filepath)
    print("numbers", numbers)
    # Conversion des positions des crochets en positions réelles
    actual_bracket = convert_to_actual_bracket(scale, bracket_pos)
    for k in range(min(len(actual_bracket), len(numbers))):
        print("( {}, {}, {} )".format(numbers[k], actual_bracket[k][0], actual_bracket[k][1]))

    time1 = time.time()
    print("time1 = ", time1)
    print("---------------------------------------------------\n\n")





if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7000)
#if __name__ == '__main__':
#    app.run(port=port)  # Démarrer l'application Flask en mode de débogage
