import os

datas = []
datas.append("/home/vrlab/hongyu/data/0712_scene_culturesquare")
datas.append("/home/vrlab/hongyu/data/0714_scene_liupanshan")
datas.append("/home/vrlab/hongyu/data/0716_scene_chengqiangmen")
datas.append("/home/vrlab/hongyu/data/0716_scene_chengqiangta")


for data in datas:
    data_path = os.path.join(data,"images")
    out_path = os.path.join(data,"output")
    os.system("python /home/vrlab/hongyu/data/gaussian_preprocess/agi2gau_update.py -out {} -img {}".format(out_path,data_path))