# Waveformer for modelling dynamical systems
The repository provides Python codes for the numerical examples illustrated in the paper ‘Waveformer for modelling dynamical systems’
Please go through the paper to understand the implemented algorithm.
Requirements:

1. Install python package pytorch, numpy, pandas, matplotlib, einops etc.

2. There are four separate folders containing data sets (data generation codes) and implementation codes, where the folders are named as
   ‘Burgers diffusin dynamics’, ‘KS equation’, ‘Allen-Cahn equation’ and Naviers stokes equation.

3. Add the data (generated data) path to load the data and use the run(Training2) file to execute program.
   
4. If you find the code helpful, please cite the paper.
@article{navaneeth2023waveformer,
  title={Waveformer for modelling dynamical systems},
  author={Navaneeth, N and Chakraborty, Souvik},
  journal={arXiv preprint arXiv:2310.04990},
  year={2023}
}
