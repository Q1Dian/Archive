# Archive

## Projects

| Project | Status |
|---------|--------|
| [EventSegmentation](./projects/EventSegmentation/) | Work in Progress |


## Environement

```
conda env create -f environment.yml
conda activate exp
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
conda install "ffmpeg<8"
pip install torchcodec --index-url=https://download.pytorch.org/whl/cu130
pip install "mamba-ssm[dev]" --no-build-isolation
pip install "causal-conv1d>=1.4.0" --no-build-isolation
```e