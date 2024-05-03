## Setup

### 1.craete a virtual environment with Python 3.10
`conda create -n whisperx_sub python==3.10`

`conda activate whisper`

### 2. Install Pytorch et al. for Linux or Windows CUDA11.8

`conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia`

*You need install Nvidia drivers for your GPU*

### 3. Install otherr dependency in requirements.txt

```bash
git clone xxxx
cd  whisperx_Sub

pip install -r requirements.txt
```