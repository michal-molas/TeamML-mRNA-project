module purge
module load Stages/2026
module load GCC

# Some base modules commonly used in AI
module load numba tqdm matplotlib
module load IPython SciPy-Stack bokeh git
module load Flask Seaborn

# ML Frameworks
module load PyTorch scikit-learn 
module load h5py
