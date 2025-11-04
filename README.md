## 1. 环境要求
同 [**DNO**](https://korrawe.github.io/dno-project/)
```python
python >= 3.7
torch >= 1.12.0
其他依赖见requirements.txt
```
### 1.1 Setup environment

Install ffmpeg (if not already installed):

```shell
sudo apt update
sudo apt install ffmpeg
```
For windows use [this](https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/) instead.


### 1.2 Install dependencies

DNO uses the same dependencies as GMD so if you already install one, you can use the same environment here.

Setup conda env:

```shell
conda env create -f environment_gmd.yml
conda activate gmd
conda remove --force ffmpeg
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
```

Download dependencies:


<summary><b>Text to Motion</b></summary>

```bash
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
```


## 2. 使用方法

### 2.1 基本生成
```bash
# 只有DPOSER损失版本
python -m sample.gen_dno_dposer_single \\
    --model_path ./save/mdm_avg_dno/model000500000_avg.pt\\
    --text_prompt "a person is jumping"

# 添加DPOSER损失版本
python -m sample.gen_dno_dposer \\
    --model_path ./save/mdm_avg_dno/model000500000_avg.pt \\
    --text_prompt "a person is jumping"
```

### 2.2 评估模式
```bash
# 运动精细化评估
python -m eval.eval_refinement_dposer \\
    --model_path ./save/mdm_avg_dno/model000500000_avg.pt \\
    --use_dposer True \\
    --dposer_weight 0.05

# 运动编辑评估
python -m eval.eval_edit_dposer \\
    --model_path ./save/mdm_avg_dno/model000500000_avg.pt \\
    --text_prompt "a person is jumping" \\
    --seed 10 \\
    --use_dposer True
```



## 3. 配置参数

### 3.1 DPoser相关参数
```python
# 在代码或配置文件中设置
USE_DPOSER = True                     # 是否启用DPoser
DPOSER_WEIGHT = 0.05                  # 正则化权重
OPTIMIZATION_STEP = 2000              # 优化步数

# DPoser具体配置
noise_opt_conf = DNOOptions(
    dposer_penalty_scale=DPOSER_WEIGHT,
    dposer_timestep_strategy="truncated",
    dposer_t_max=0.15,
    dposer_t_min=0.05,
    dposer_t_fixed=0.1,
    dposer_use_snr_weighting=True,
)
```

### 3.2 常用任务类型
- motion_editing: 运动编辑
- pose_editing: 姿势编辑
- trajectory_editing: 轨迹编辑
- motion_projection: 运动投影
- motion_blending: 运动融合
- motion_inbetweening: 运动填充

## 4. 输出结果
生成的结果保存在：
```
./results/{model_name}/samples_{seed}_{prompt}/{task_name}_{mode}/
```







## 5. DNO & DNO_single 文件说明

### 5.1 dno.py
加入dposer相关loss。

### 5.2 dno_single.py
DNO_single实现了单步去噪DPoser（类似原版DPoser），提供更高效的计算。
