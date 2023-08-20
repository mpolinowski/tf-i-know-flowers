import json
import numpy as np
from PIL import Image
import requests


labels = ['Gladiolus', 'Adenium', 'Alpinia_Purpurata', 'Alstroemeria', 'Amaryllis', 'Anthurium_Andraeanum', 'Antirrhinum', 'Aquilegia', 'Billbergia_Pyramidalis', 'Cattleya', 'Cirsium', 'Coccinia_Grandis', 'Crocus', 'Cyclamen', 'Dahlia', 'Datura_Metel', 'Dianthus_Barbatus', 'Digitalis', 'Echinacea_Purpurea', 'Echinops_Bannaticus', 'Fritillaria_Meleagris', 'Gaura', 'Gazania', 'Gerbera', 'Guzmania', 'Helianthus_Annuus', 'Iris_Pseudacorus', 'Leucanthemum', 'Malvaceae', 'Narcissus_Pseudonarcissus', 'Nerine', 'Nymphaea_Tetragona', 'Paphiopedilum', 'Passiflora', 'Pelargonium', 'Petunia', 'Platycodon_Grandiflorus', 'Plumeria', 'Poinsettia', 'Primula', 'Protea_Cynaroides', 'Rose', 'Rudbeckia', 'Strelitzia_Reginae', 'Tropaeolum_Majus', 'Tussilago', 'Viola', 'Zantedeschia_Aethiopica']


url1 = 'http://localhost:8501/v1/models/efficientv2b0_model:predict'
url2 = 'http://localhost:8501/v1/models/mobilenetv3S_model:predict'
url3 = 'http://localhost:8501/v1/models/vit_model:predict'


test_img1 = "/opt/app/snapshots/Viola_Tricolor.jpg"
test_img2 = "/opt/app/snapshots/Water_Lilly.jpg"
test_img3 = "/opt/app/snapshots/Strelitzia.jpg"


with Image.open (test_img2) as im:
        preprocess_img = im.resize((224, 224))

batched_img = np.expand_dims(preprocess_img, axis=0)
batched_img = np.float32(batched_img)

data = json.dumps(
    {"signature_name": "serving_default", "instances": batched_img.tolist()}
)


def predict_rest(json_data, url):
    json_response = requests.post(url, data=json_data)
    response = json.loads(json_response.text)
    rest_outputs = np.array(response["predictions"])
    return rest_outputs


# get prediction from efficientv2b0_model
rest_outputs = predict_rest(data, url1)
index = np.argmax(rest_outputs, axis=-1)[0]  # Index with highest prediction

print("Prediction Results: EfficientV2B0")
print("Class probabilities: ", rest_outputs)
print("Predicted class: ", labels[index])
percentage = round((rest_outputs[0][index]*100), 3)
print(f'Certainty:  {percentage} %')


# get prediction from mobilenetv3S_model
rest_outputs = predict_rest(data, url2)
index = np.argmax(rest_outputs, axis=-1)[0]  # Index with highest prediction

print("Prediction Results: MobileNetV3S")
print("Class probabilities: ", rest_outputs)
print("Predicted class: ", labels[index])
percentage = round((rest_outputs[0][index]*100), 3)
print(f'Certainty:  {percentage} %')


# get prediction from vit_model
rest_outputs = predict_rest(data, url3)
index = np.argmax(rest_outputs, axis=-1)[0]  # Index with highest prediction

print("Prediction Results: ViT")
print("Class probabilities: ", rest_outputs)
print("Predicted class: ", labels[index])
percentage = round((rest_outputs[0][index]*100), 3)
print(f'Certainty:  {percentage} %')
