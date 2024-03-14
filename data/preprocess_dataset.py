import json
import shutil
import os

train_json = './dataset/annotations/new_train.json'
test_json = './dataset/annotations/new_test.json'
val_json = './dataset/annotations/new_valid.json'

image_dir = './dataset/image'
os.makedirs(image_dir, exist_ok=True)

with open(train_json) as f:
    train_data = json.load(f)

with open(test_json) as f:
    test_data = json.load(f)

with open(val_json) as f:
    val_data = json.load(f)


current_file_path = os.path.abspath(__file__).split('/')[:-1]
current_file_path = '/'.join(current_file_path)+'/dataset/image'
print("current absolute path:", current_file_path)


def update_annotation(data, json_file):
    updated_data = []
    for item in data:
        file_name = item[0].split('/')[-1]
        new_item = item
        new_item[0] = os.path.join(current_file_path, file_name)
        updated_data.append(new_item)
    with open(json_file, 'w') as f:
        json.dump(updated_data, f)

new_train_json = './dataset/annotations/new_train.json'
update_annotation(train_data, new_train_json)

new_test_json = './dataset/annotations/new_test.json'
update_annotation(test_data, new_test_json)

new_val_json = './dataset/annotations/new_valid.json'
update_annotation(val_data, new_val_json)