# BiasMind

This project contains the data and code for **Discover Cognitive Biases via Neural-symbolic Induction**.

## Project Overview

This project discovers cognitive biases through neural-symbolic induction methods BiasMind.
Due to file size limitations, the Our NewsCog dataset is available at the following link. https://drive.google.com/drive/folders/1SHwZ13HzBehWKt4HmK1Q6_uXNsIsjg1U?dmr=1&ec=wgc-drive-hero-gotoPlease download it and place it in the data/ folder.


## Environment Setup

### Dependencies Installation

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies include:
- PyTorch 2.0.1
- transformers
- scikit-learn
- numpy
- pandas
- and more (see `requirements.txt` for details)

## Configuration

### Path Replacement

**Important:** Before running the code, you need to replace all `/path/code/` paths in the code with your own local paths.

Files that need path replacement include:

1. **drive_dummy.py**
   - `--data_path`: Default value is `/path/code/BiasMind/data`
   - `--save_path`: Default value is `/path/code/BiasMind/data/cognitive/report_all11.json`
   - `--save_all_path`: Default value is `/path/code/BiasMind/data/cognitive/save`
   - `--gqfile`: Default value is `/path/code/BiasMind/data/cognitive/...`

2. **drive_prune.py**
   - `--data_path`: Default value is `/path/code/BiasMind/data/`
   - `--save_path`: Default value is `/path/code/BiasMind/data/cognitive/report.json`
   - `--save_all_path`: Default value is `/path/code/BiasMind/data/save`
   - `--gqfile`: Default value is `/path/code/BiasMind/data/cognitive/...`

3. **models/qa_t5.py**
   - `HF_HOME`: `/path/code/BiasMind/`
   - `DEEPSEEK_PATH`: `/path/code/Pre_model/`
   - `FT5_PATH`: `/path/code/Pre_model/`
   - `Llama_PATH`: `/path/code/Pre_model/`

4. **rule_parase.py**
   - `file_path`: `/path/code/BiasMind/rules.txt`

5. **run.sbatch** (if using SLURM)
   - `/path/code/` paths in the script

### Quick Replacement Method

You can use the following commands to batch replace paths (modify according to your actual situation):

```bash
# Linux/Mac
find . -type f -name "*.py" -exec sed -i 's|/path/code/|/your/local/path/|g' {} \;

# Windows PowerShell
Get-ChildItem -Recurse -Filter *.py | ForEach-Object { (Get-Content $_.FullName) -replace '/path/code/', '/your/local/path/' | Set-Content $_.FullName }
```

## Running the Code

**Note:** Before running, please ensure:
1. All `/path/code/` paths have been replaced with local paths
2. Path parameters in `run.sh` have been modified (`--data_path` and `--gqfile`)
3. All dependencies have been installed

### Method 1: Using run.sbatch (SLURM Cluster)

If running on a SLURM cluster, you can use:

```bash
sbatch run.sbatch
```

### Method 2: Running Python Scripts Directly

#### Training Model (drive_dummy.py)

```bash
python drive_dummy.py \
    --dataset_name cognitive \
    --cog_name 结果偏差 \
    --data_path /your/local/path/BiasMind/data \
    --gqfile /your/local/path/BiasMind/data/cognitive/your_gq_file.json \
    --n_out 15 \
    --batchsize 16 \
    --lr 1e-3 \
    --n_epoch 15 \
    --device cuda
```

#### Model Pruning and Post-processing (drive_prune.py)

```bash
python drive_prune.py \
    --dataset_name cognitive \
    --data_path /your/local/path/BiasMind/data/ \
    --gqfile /your/local/path/BiasMind/data/cognitive/your_gq_file.json \
    --best_target_ckpoint bestmodel \
    --best_dir your_model_directory \
    --device cuda
```

## Project Structure

```
BiasMind/
├── drive_dummy.py          # Main training script
├── drive_prune.py          # Model pruning and post-processing script
├── focal_loss.py           # Focal Loss implementation
├── rule_parase.py          # Rule parsing script
├── run.sbatch              # SLURM batch script
├── requirements.txt        # Python dependencies list
├── models/                  # Model-related code
│   ├── qa_t5.py           # T5 question-answering model
│   ├── dnf.py             # DNF model
│   └── ...
├── utils/                   # Utility functions
│   ├── data_reading.py    # Data reading
│   ├── evaluation.py      # Evaluation metrics
│   ├── components/        # Component modules
│   │   ├── dnf_layer.py   # DNF layer implementation
│   │   └── ...
│   └── ...
└── README.md               # This file
```

## Supported Cognitive Bias Types

The project supports training for the following cognitive bias types:

- Confirmation Bias (确认偏差)
- Availability Heuristic (可用性启发式)
- Stereotype (刻板印象)
- Halo Effect (光环效应)
- Authority Bias (权威偏见)
- Framing Effect (框架效应)
- Bandwagon Effect (从众效应)
- In-group Favoritism (群体内偏爱)
- Contrast Effect (对比效应)
- Overconfidence Effect (过度自信效应)
- Loss Aversion (损失厌恶)
- Outcome Bias (结果偏差)
- Hindsight Bias (后见之明偏差)

## Main Parameters

### drive_dummy.py Main Parameters

- `--cog_name`: Cognitive bias name (see list above)
- `--dataset_name`: Dataset name (default: cognitive)
- `--data_path`: Data path
- `--gqfile`: General question file path
- `--n_out`: Number of output classes (2 or 15)
- `--num_conjuncts`: Number of conjuncts (default: 50)
- `--batchsize`: Batch size (default: 16)
- `--lr`: Learning rate (default: 1e-3)
- `--n_epoch`: Number of training epochs (default: 15)
- `--type_of_logic_model`: Logic model type (gnn_logic, hgt_logic, logic, etc.)
- `--device`: Device (cuda or cpu)

### drive_prune.py Main Parameters

- `--best_target_ckpoint`: Model checkpoint filename to load
- `--best_dir`: Model save directory
- `--prune_epsilon`: Performance drop threshold allowed for pruning
- Other parameters are similar to `drive_dummy.py`

## Notes

1. **Path Configuration**: Ensure all paths have been correctly replaced with local paths
2. **Data Files**: Ensure data files (e.g., gq_file) exist and paths are correct
3. **Model Files**: If using pre-trained models, ensure model paths are correct
4. **GPU Memory**: Adjust the `batchsize` parameter according to GPU memory
5. **API Keys**: If using OpenAI API or other APIs, configure them in `utils/api_key.txt`

## License

Please refer to the project license file.

## Contact

For questions, please refer to the project documentation or contact the project maintainer:

**Email:** dongyiqi@tju.edu.cn

## Acknowledgment
Our implementation is mainly based on follows. Thanks for their authors. https://github.com/less-and-less-bugs/Trust_TELLER

