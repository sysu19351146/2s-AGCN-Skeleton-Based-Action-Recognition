2s-AGCN-Skeleton-Based-Action-Recognition
==

版本
==
pytorch 1.8.1  <br>
注：需要安装tensorboardX库：    `pip install tensorboardX`

数据准备
==
* npy数据转json文件  <br>
      `python numpy2json.py`     
* 数据处理  <br>
      `python data_gen/kinetics_gendata.py`
* 生成bone的数据  <br>
      `python data_gen/gen_bone_data.py`
 
训练和测试：
==

* 训练 <br>
  `python main.py --config ./config/nturgbd-cross-view/test_joint.yaml`<br>
  `python main.py --config ./config/nturgbd-cross-view/test_bone.yaml` <br>

* 测试  <br>
  `python main.py --config ./config/nturgbd-cross-view/train_joint.yaml`<br>
  `python main.py --config ./config/nturgbd-cross-view/train_bone.yaml` <br>


