import json
import os
with open("transforms_test.json","r") as f:
    test_data = json.load(f)
with open("transforms_train.json","r") as f:
    train_data = json.load(f)
with open("transforms_val.json","r") as f:
    val_data = json.load(f)
for i in range(len(train_data["frames"])):
    train_data["frames"][i]["depth_path"] = os.path.join("disp_png", train_data["frames"][i]["file_path"].split("/")[-1])
for i in range(len(val_data["frames"])):
    val_data["frames"][i]["depth_path"] = os.path.join("disp_png", val_data["frames"][i]["file_path"].split("/")[-1])
for i in range(len(test_data["frames"])):
    test_data["frames"][i]["depth_path"] = os.path.join("disp_png", test_data["frames"][i]["file_path"].split("/")[-1])
with open("transforms_test.json","w") as f:
    json.dump(test_data,f,indent=4)
with open("transforms_train.json","w") as f:
    json.dump(train_data,f,indent=4)
with open("transforms_val.json","w") as f:
    json.dump(val_data,f,indent=4)