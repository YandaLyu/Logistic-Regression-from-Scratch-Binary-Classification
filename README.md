# Logistic-Regression-from-Scratch-Binary-Classification

Yanda Lyu | University of San Francisco

Project Overview
This project implements logistic regression for binary classification from scratch (using only NumPy, no sklearn models), and evaluates its performance on the Breast Cancer dataset and the MNIST "5 vs not 5" task. Both gradient descent and stochastic gradient descent (SGD) are implemented for cost minimization.

Technical Workflow
1. Implementation Details
Sigmoid Activation: Implemented for model output.

Cost Function: Binary cross-entropy loss.

Gradient Calculation: Manual computation for both GD and SGD.

Optimization:

Batch Gradient Descent with tunable learning rate and max iterations.

Stochastic Gradient Descent (SGD) for scalable training.

2. Data Loading & Preprocessing
Breast Cancer Dataset:
Loaded via sklearn.datasets.load_breast_cancer(), standardized features, split into training/validation sets.

MNIST Dataset:
Loaded via sklearn.datasets.fetch_openml('mnist_784').

Reformulated as a binary classification: "5" vs. "not 5".

Used only SGD due to dataset size (60000 training, 10000 validation).

3. Experimentation & Tuning
Learning Rate Search:
Ran experiments with multiple learning rates for both GD and SGD.

Convergence Analysis:
Plotted cost vs. iteration (GD) and cost vs. epoch (SGD) for all rates.

Comparison:
On breast cancer data, compared convergence speed and quality for best learning rate in GD vs. SGD.

4. Evaluation & Visualization
Accuracy:
Reported classification accuracy on the validation set for each dataset/task.

Confusion Visualization (MNIST):
Displayed the 8 most "confused" validation images—those for which the model was confident but incorrect—with analysis.

Convergence Plots:
All experiments include cost function curves to show training progress and convergence.

Key Findings
Learning Rate Sensitivity:
Convergence and final accuracy are highly sensitive to learning rate selection for both GD and SGD.

Convergence:
SGD typically converges faster per epoch but can have higher variance; both methods reach similar minima with good tuning.

Model Limitations:
MNIST 5-vs-not-5 reveals that logistic regression struggles with ambiguous hand-written digits, as visualized in confused images.
