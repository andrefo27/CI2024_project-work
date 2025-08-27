## CI2024_project-work
This project explores **symbolic regression** using **genetic programming**.  
The main objective is to evolve compact and accurate mathematical expressions that approximate the relationship between input variables (**x**) and an output variable (**y**).

#### Repository Structure

- **data/**  
    - Contains the eight datasets used for the mathematical approximation.

- **src/**  
    - Contains `algorithm.ipynb`, where the entire logic for solving the symbolic regression problem has been defined.

- **s324581.py**  
    - File that collects the selected formulas found for each problem. These are the functions obtained executing the GP algorithm with 100% of the entire dataset.

- **res/**  
    - It contains all the evaluation formulas and their results for each dataset. You can experiment with the evaluations using the `evaluation.py` file.
    - It contains the `s324581_datasplit.py` with the functions obtained executing the GP algorithm with 60% of the entire dataset. For the final evaluation, you should use also this file to verifiy how the generalization works on the broader dataset

- **report_CI2024_project-work_s32458.pdf**
    - detailed description of the project under examination and of the work carried out in the computational intelligence course

#### Algorithm overview

1. **Experimental Objective**  
   The goal is to evolve candidate expressions (treated as chromosomes) across multiple generations, balancing accuracy and simplicity. A split-dataset approach is used: models are trained on a training set and validated on a separate dataset to avoid overfitting.

2. **Representation & Evolution**  
   - **Expression Trees**: Solutions are represented as trees, where internal nodes are functions (e.g., `+`, `sin`) and leaves are variables (`x0`, `x1`, …) or constants.  
   - **Genetic Operators**:  
     - **Crossover**: Subtrees are exchanged between parents to combine traits.  
     - **Mutations**: Multiple mutation types are used to maintain diversity and explore the search space, including subtree, point, hoist, expansion, and collapse mutations.  

3. **Fitness Function**  
   Solutions are evaluated with a multi-objective score combining:  
   - **Mean Squared Error (MSE)** – main accuracy metric  
   - **Size Penalty** – discourages overly complex formulas  
   - **Variable Usage Penalty** – promotes balanced use of inputs  
   - **Ratio Penalty** – discourages over-reliance on constants  

4. **Evolutionary Loop**  
   - **Initial Population**: Randomly generated expressions  
   - **Iterative Evolution**: Selection, crossover, and mutation over generations  
   - **Elitism**: Best solutions preserved to the next generation  
   - **Immigration**: Random new solutions injected to escape local optima 
   - **Forced Mutation**: Every N generations, all the best individuals (elite group) are subjected to an additional mutation
   - **Stagnation Recovery Strategy**: When the population shows no improvement for a fixed number of generations
   - **Final Evaluation**: Best expression tested on the validation set to ensure generalization  


#### Mean Squared Error (MSE) results
You can experiment with fitness in the `test/evaluation.py` file.

MSE Results for functions in `s324581_datasplit.py`. These are the functions obtained executing the GP algorithm with 60% of the entire dataset.
Comparison with the validation set (as you can see in the report: `report_CI2024_project-work_s324581.pdf`) shows a well generalization

Fitness was calculated for the entire dataset

| Problem | Function | MSE          |
|---------|----------|--------------|
| 1       | f1       | 0.000000     |
| 2       | f2       | 4021359888365.140625 |
| 3       | f3       | 0.000434     |
| 4       | f4       | 0.000013     |
| 5       | f5       | 0.000000     |
| 6       | f6       | 0.000000     |
| 7       | f7       | 330.585762   |
| 8       | f8       | 2161.756642  |


MSE Results for functions in `s324581.py`. These are the functions obtained executing the GP algorithm with 100% of the entire dataset.
We don't have a validation set to test if generalize well

| Problem | Function | MSE          |
|---------|----------|--------------|
| 1       | f1       | 0.000000     |
| 2       | f2       | 2796718779356.910645 |
| 3       | f3       | 0.262319     |
| 4       | f4       | 0.078527     |
| 5       | f5       | 0.000000     |
| 6       | f6       | 0.0000003    |
| 7       | f7       | 327.512789   |
| 8       | f8       | 76.3244082  |