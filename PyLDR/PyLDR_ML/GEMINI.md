# PyLDR_ML: Machine Learning Control System

PyLDR_ML is a hybrid hardware-software project focused on implementing a control system using Machine Learning. It integrates Arduino-based hardware control for sensors and actuators with Python-based data acquisition, system identification, and controller optimization.

## Project Architecture

The project is highly modular, following a "folder-as-module" pattern with primary implementations in `index.py`, `index.h`, or `index.ts`.

- **Aquisition/**: Contains both Arduino (`Aquisition.ino`) and Python (`index.py`) code for data collection. The hardware generates controlled stimulus (e.g., random servo movements) and streams JSON data over Serial, while the Python script captures this data into CSV format for training.
- **Hardware/**: The core Arduino codebase. It uses an OOP approach with a `Device` class to abstract sensors (Ultrasonic), actuators (Servo), and system tasks.
- **Model/**: The machine learning and control theory workspace.
  - `1 - Analysis.ipynb`: Data exploration, cleaning, and preprocessing.
  - `2 - System.ipynb`: System identification to create a mathematical model of the physical system.
  - `3 - Controller.ipynb`: Design and tuning of the control logic.
  - **GridSearch/**: Custom implementation for hyperparameter optimization using time-series cross-validation.
  - **Metrics/**: Tools for evaluating model performance through cross-validation and statistical analysis.
- **Dataset/**: Repository for raw and processed experimental data (`data.csv`, `data_cleaned.csv`).
- **Controller/**: Likely contains the final control implementation to be deployed back to the hardware.

## Building and Running

### Hardware (Arduino)
1. Open the project in the Arduino IDE or VS Code with the Arduino extension.
2. Select the appropriate board and port.
3. Upload `Hardware/Main.ino` for the main operation or `Aquisition/Aquisition.ino` for data collection.
4. **Note**: Serial communication typically operates at `115200` or `9600` baud (check individual `.ino` files).

### Software (Python)
1. **Environment**: This project uses `conda`. Activate the base environment:
   ```bash
   conda activate base
   ```
2. **Dependencies**: Ensure the following libraries are installed:
   ```bash
   pip install pandas pyserial matplotlib scikit-learn joblib control
   ```
3. **Data Acquisition**:
   ```bash
   python Aquisition/index.py
   ```
4. **Model Development**: Open the Jupyter notebooks in the `Model/` directory to follow the analysis and design pipeline.

## Development Conventions

- **Language**: All code and documentation are strictly in English.
- **Naming**:
  - `PascalCase` for Classes.
  - `camelCase` for variables, methods, instances, and arguments.
- **Paradigm**: Strictly Object-Oriented Programming (OOP).
- **Structure**: Treat folders as discrete components. Primary logic should reside in an `index` file within the directory.
- **Python Specifics**: Do not use type hints or annotations.
- **Quality**: Prioritize clean, self-documenting code over excessive comments.
