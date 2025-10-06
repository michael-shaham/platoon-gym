# Platoon Gym

Gym environment for prototyping, evaluating, and benchmarking platooning 
algorithms. This package uses our `i24_forecasting` package to obtain virtual 
leader trajectories and trained forecasting models.

Setup using conda (**recommended**):

```bash
mkdir convoy && cd convoy
conda create -n convoy python=3.10
conda activate convoy
git clone git@github.com:RIVeR-Lab/platoon-gym.git
git clone git@github.com:RIVeR-Lab/i24_forecasting.git
cd i24_forecasting
pip install -e .
cd ../platoon-gym
pip install -e .
```

Setup using venv:

```bash
mkdir convoy
python3 -m venv .convoy_env
source .convoy_env/bin/activate
git clone git@github.com:RIVeR-Lab/platoon-gym.git
git clone git@github.com:RIVeR-Lab/i24_forecasting.git
cd i24_forecasting
pip3 install -e .
cd ../platoon-gym
pip3 install -e .
```