2s-AGCN-Skeleton-Based-Action-Recognition
==

版本
--
pytorch 1.8.1  <br>
注：需要安装tensorboardX库：    `pip install tensorboardX`

数据准备
--
* npy数据转json文件  <br>
      `python numpy2json.py`     
* 数据处理  (注意要使用cd改变当前路径)<br>
      `cd data_gen`<br>
      `python kinetics_gendata.py`<br>
      `cd ..`<br>
* 生成bone的数据 (注意要使用cd改变当前路径)  <br>
      `cd data_gen`<br>
      `python gen_bone_data.py`<br>
      `cd ..`<br>
 
训练和测试：
--

* 训练 <br>
  `python main.py --config ./config/kinetics-skeleton/test_joint.yaml`<br>
  `python main.py --config ./config/kinetics-skeleton/test_bone.yaml` <br>

* 测试  <br>
  `python main.py --config ./config/kinetics-skeleton/train_joint.yaml`<br>
  `python main.py --config ./config/kinetics-skeleton/train_bone.yaml` <br>
* 整合  <br>
  `python ensemble.py`     
  
* 整个测试流程: <br>
  先分别test： <br>
  `python main.py --config ./config/kinetics-skeleton/test_joint.yaml`<br>
  `python main.py --config ./config/kinetics-skeleton/test_bone.yaml` <br>
  再整合： <br>
  `python ensemble.py` <br>
  
参考文献：
--
[1] Shi L , Zhang Y , Cheng J , et al. Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition[J]. 2018. <br>
[2] Zhang P , Lan C , Zeng W , et al. Semantics-Guided Neural Networks for Efficient Skeleton-Based Human Action Recognition[C]// 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2020.

