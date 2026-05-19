# PyLDR: Multi-Strategy Control System Project

PyLDR is a research-oriented project focused on developing and comparing different control strategies for a hardware system (likely involving Light Dependent Resistors - LDRs). It implements three distinct approaches to control: Machine Learning (ML), Reinforcement Learning (RL), and traditional PID control.

## Project Structure

The project is organized into three main modules, each representing a different control strategy:

- **PyLDR_ML/**: Implements control using standard Machine Learning regression models (Linear Regression, SVR, Random Forest, etc.).
- **PyLDR_RL/**: Explores Reinforcement Learning approaches, including neural network-based selectors.
- **PyLDR_PID/**: Implements traditional Proportional-Integral-Derivative control.

### Shared Module Architecture
Each module follows a consistent internal structure:
- **Dataset/**: Stores experimental data (`data.csv`, `data_cleaned.csv`).
- **Model/**: The core development workspace.
  - `1 - Analysis.ipynb`: Data cleaning and exploratory analysis.
  - `2 - System.ipynb`: System identification and mathematical modeling.
  - `3 - Controller.ipynb`: Controller design, tuning, and evaluation.
  - **GridSearch/**: Custom hyperparameter optimization tools.
  - **Metrics/**: Evaluation utilities (Cross-validation, Plotting).
  - **Selector/**: Model selection logic for testing different algorithms.

## Technologies

- **Languages**: Python (Analysis & Control), C++ (Arduino Hardware Interface).
- **Libraries**: `scikit-learn`, `pandas`, `matplotlib`, `pyserial`, `control`, `joblib`.
- **Environment**: Managed via `conda` (base environment).
- **Hardware**: Arduino-based sensors and actuators (interfaced via Serial).

## Building and Running

### Environment Setup
1. Activate the base conda environment:
   ```bash
   conda activate base
   ```
2. Install required Python dependencies:
   ```bash
   pip install pandas pyserial matplotlib scikit-learn joblib control
   ```

### Development Workflow
The development process for each control strategy follows a sequential pipeline within the `Model/` directory:
1. **Analysis**: Run `1 - Analysis.ipynb` to process raw sensor data.
2. **System Identification**: Run `2 - System.ipynb` to derive the plant model.
3. **Controller Design**: Run `3 - Controller.ipynb` to implement and tune the specific control logic (ML, RL, or PID).

## Development Conventions

- **Language**: English only for code and documentation.
- **Naming**: `PascalCase` for classes; `camelCase` for variables and methods.
- **Paradigm**: Strictly Object-Oriented Programming (OOP).
- **Modularity**: "Folder-as-module" pattern; main logic resides in `index` files within directories.
- **Python**: No type hints or annotations are used in this project.
