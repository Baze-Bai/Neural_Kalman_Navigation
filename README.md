# Neural-Kalman navigation reproduction and exploration

A source-only reproduction/extension workspace for GNSS/INS precision-agriculture navigation. The original neural-Kalman method, paper, dataset and published metrics belong to the upstream researchers, not to this portfolio owner.

## Upstream attribution

Original code: [nesl/agrobot](https://github.com/nesl/agrobot). The BSD 3-Clause license and source attribution are preserved. Authors of the original ICRA 2023 work include Yayun Du, Swapnil Sayan Saha, Sandeep Singh Sandha, Arthur Lovekin, Jason Wu, S. Siddharth, Mahesh Chowdhary, Mohammad Khalid Jawed and Mani Srivastava.

## Included files

`Neural Kalman IMU GNSS Fusion/` contains filter/trajectory utilities, training, fusion, fine-tuning and ablation notebooks, plus a local `Spike_TCN.ipynb` exploration notebook. Their presence is not a claim that every component was originally authored here or that the exploration improves the upstream method.

Technologies: Python, TensorFlow, Kalman filtering, NumPy/SciPy, and an exploratory PyTorch/spiking-neural-network notebook. Review notebook imports for additional exploration dependencies. The retained upstream requirements specify the original TensorFlow environment; that environment is distinct from the exploratory notebook's requirements.

Download datasets and pretrained weights directly from the upstream source under its terms. They are deliberately not included. Place required upstream assets in the matching relative paths and review environment-specific settings. Notebook outputs and execution metadata were removed. Source syntax and notebook structure were checked; navigation accuracy and model training were not revalidated for publication.
