import tensorflow.keras.models.model_from_json as model_from_json

curr_model_weights = ''
new_model_architecture = '.json'

# TODO - modify the JSON file to get something that fits (using X-CUBE AI)
model = model_from_json(new_model_architecture)
model.load_weights(curr_model_weights, by_name=True)

model.save('newest_model.h5')