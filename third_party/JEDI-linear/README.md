# Reproduction Instructions

Install environment

```bash
conda env create -f environment.yml
conda activate jedi-linear
```

### Download and prepare dataset

```bash
bash prepare_dataset.sh
```


### Obtaining the models

The models shown in the tables in the papers are included in `official_models.tar.gz`.
You can extract them with:

```bash
tar -xvf official_models.tar.gz
```

If you want to retrain the models, you can do so with the following commands.
This will launch a whole Pareto scan, so many models will be saved.

```bash
KERAS_BACKEND=jax python jet_classifier.py -c configs/<config_file> -r train
```

### Evaluation on test set, convert to Verilog

The outputs are already included in the `official_models.tar.gz`, but you can validate them with:

```bash
KERAS_BACKEND=jax python jet_classifier.py -c configs/<config_file> -r test verilog
```

The Verilator may require a newer C++ compiler. We tested our code with g++ 15.1.1.

### Synthesis

Due to size consideration, we removed the Vivado project files, but only included the gererated reports.

```bash
cd <output_directory>/<model_directory>/da4ml_verilog_prjs/<verilog_project>
vivado -mode batch -source build_prj.tcl

# Starting v0.5.x, the da4ml generated projects layout changed, and the synthesis script is now named build_vivado_prj.tcl
# vivado -mode batch -source build_vivado_prj.tcl
```

### Generate json report

Using the included `load_summary.py` script in the tarball, you can generate the json report from all the synthesis results:

```bash
cd <output_directory>
for p in *-feature*; do
    for N in 8 16 32 64 128; do
        name=$(basename $p)
        for f in $p/*$N; do python3 load_summary.py -e $f/test_acc.json $f/da4ml_verilog_prjs/* -o summary/$N-particle-$name.json; done
    done
done
```

# Citation

```{=latex}
 @inproceedings{jedi-linear,
  title={JEDI-linear: Fast and Efficient Graph Neural Networks for Jet Tagging on FPGAs},
  author={Que, Zhiqiang and Sun, Chang and Paramesvaran, Sudarshan and Clement, Emyr and Karakoulaki, Katerina and Brown, Christopher and Laatu, Lauri and Cox, Arianna and Tapper, Alexander and Luk, Wayne and Spiropulu, Maria},
  booktitle={2025 International Conference on Field Programmable Technology (FPT)},
  year={2025},
  organization={IEEE}
}
```


# JEDI-linear - General Instructions


This repository contains the code for the paper "JEDI-linear: Fast and Efficient Graph Neural Networks for Jet Tagging on FPGAs" (https://arxiv.org/abs/2508.15468). The code can be run as follows:

1. Clone the repository, install the dependencies. Also install one of the backends for training (`jax`, `tensorflow`, or `pytorch`)
   ```bash
   pip install -r requirements.txt
   pip install <your_backend>
   ```
2. Download the dataset from https://zenodo.org/records/3602260
3. Extract the training and testing (the validation split downloaded), prepare them with
   ```bash
   python prepare_dataset.py -i /tmp/<train/validation>/ -o /tmp/<train/test>.h5 -j <n_processes>
   ```
   Place both `train.h5` and `test.h5` in the same directory, e.g., `/tmp/jet_data/`.
4. Modifying the configs to have the `datapath` point to the dataset directory, and change the output directory `save_path` if needed.
5. Run the training script:
   ```bash
   KERAS_BACKEND=<YOUR_BACKEND> python jet_classifier -c <CONFIG_FILE> -r train test verilog
   ```
   where `<YOUR_BACKEND>` can be `jax`, `tensorflow`, or `torch` depending on the backend you installed.
   The configs are located in `configs/`.
   The `-n$number` part of the config file is the maximum number of particles to be used; `-3` means only `pt, eta, phi` are used, otherwise all 16 features are used; `uq1` means the network is uniformly quantized over the particle dimension and is permutation-invariant.

