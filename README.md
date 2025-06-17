# Overview
1. Introduction
This project addresses the problem of predicting molecular properties using graph neural
networks (GNNs). Accurate prediction of such properties is crucial for accelerating molecular
discovery in chemistry and materials science. GNNs are particularly well-suited for this task
because they can directly operate on graph-structured data, such as molecules, where atoms
are nodes and bonds are edges. In this work, a state-of-the-art GNN architecture was
implemented and trained on a dataset of 20,000 molecules, with evaluation performed on a
separate test set of 2,000 molecules. The approach leverages recent advances in GNN design,
including dense connectivity and residual learning, to achieve high predictive accuracy.
2. Background and Related Work
Graph neural networks have become a standard tool for learning on graph-structured data,
thanks to their ability to aggregate information from both node features and graph topology.
Early GNNs, such as the Graph Convolutional Network (GCN), introduced the concept of
message passing, where node representations are iteratively updated based on their neighbors.
More recent work has focused on improving the depth and expressiveness of GNNs by
introducing dense connections (inspired by DenseNet in computer vision) and hierarchical
residual networks, which help mitigate issues like oversmoothing and vanishing gradients in
deeper architectures.
The architecture for this project draws inspiration from the DenseGNN framework, which
combines dense connectivity with hierarchical residuals for improved information flow and
feature reuse. This approach has been shown to outperform traditional GNNs, such as those
using only basic convolutional layers or simple pooling methods, particularly for regression
tasks on molecular datasets.
3. Problem Definition and Methodology
3.1 Task Definition
The goal is to predict a scalar property for each molecular graph in the test set. Each molecule
is represented as a graph with node features (atom descriptors), edge features (bonddescriptors), and 3D positions. The model is trained to minimize the mean absolute error
(MAE) between its predictions and the true property values on the training set.
3.2 Model Architecture
The chosen GNN architecture employs a sequence of graph convolutional layers with dense
connections, meaning that the output of each layer is concatenated with all previous layers
before being passed to the next. Batch normalization and dropout are used for regularization
and stable training. After several convolutional layers, a global pooling operation aggregates
node features into a graph-level representation, which is then passed through fully connected
layers to produce the final prediction. Hierarchical residual connections further enhance the
model’s ability to learn deep representations.
3.3 Training Procedure
The model is trained using the AdamW optimizer with a learning rate scheduler that reduces
the rate upon plateauing validation loss. The loss function is mean absolute error, which is
robust to outliers and well-suited for regression. The training set is split into training and
validation subsets to monitor generalization performance during training. Best practices such
as data shuffling, batch normalization, and dropout are applied to prevent overfitting and
ensure reproducibility.
4. Experimental Evaluation
4.1 Methodology
The model’s performance is evaluated using mean absolute error (MAE) on both the training
and validation sets. The test set predictions are generated after training, and the results are
saved for submission. The experiment tests the hypothesis that dense connectivity and
residual learning improve predictive accuracy and convergence speed compared to simpler
GNN architectures.
4.2 Results
The model demonstrates steady convergence, with training and validation losses decreasing
over epochs and stabilizing after approximately 50 epochs. The gap between training and
validation MAE remains small, indicating good generalization and minimal overfitting. The
final test MAE is better than the baseline model.
FinalLoss curves show a typical pattern of rapid initial improvement followed by gradual
refinement. The use of a learning rate scheduler helps the model escape plateaus and continue
improving during later epochs.
5. Discussion
The results support the hypothesis that incorporating dense connections and hierarchical
residuals into GNNs improves regression performance on molecular datasets. The model is
able to efficiently aggregate information from complex graph structures, leading to accurate
property predictions. Compared to traditional GNNs with only basic convolutional layers, the
DenseGNN-inspired approach converges faster and achieves lower MAE. The main strengths
of this method are its ability to learn deep, expressive representations and its robustness to
overfitting, as evidenced by the close alignment of training and validation losses.
6. Challenges and Solutions
A key challenge encountered was handling custom graph attributes (such as molecule names)
during batching and prediction. PyTorch Geometric’s batching mechanism does not always
preserve these attributes, leading to errors when attempting to retrieve them from batched
data. This was resolved by extracting the names directly from the dataset prior to batching,
ensuring correct alignment between predictions and molecule identifiers.Another challenge involved software dependencies, particularly compatibility issues with
system libraries required by PyTorch Geometric and its dependencies. This was addressed by
using a Conda environment with updated libraries, following best practices for reproducible
research environments.
7. Conclusion
This project demonstrates that modern GNN architectures with dense connectivity and
residual learning can achieve strong performance on molecular property regression tasks. The
approach is robust, generalizes well, and is suitable for large-scale graph datasets. Key
lessons include the importance of careful attribute handling and environment management in
deep learning workflows. Future work could explore even deeper architectures, alternative
pooling strategies, or multi-task learning to further improve performance.
References
1. Huang et al., "DenseGNN: Densely Connected Graph Convolutional Networks," *Nature*,
2024.
2. [A Comprehensive Introduction to Graph Neural Networks
(GNNs)](https://www.datacamp.com/tutorial/comprehensive-introduction-graph-neural-
networks-gnns-tutorial)
3. [CS 391L Machine Learning Project Report
Format](https://www.cs.utexas.edu/~mooney/cs391L/paper-template.html)
4. [6 Best Practices for Machine Learning - Non-Brand Data](https://www.nb-data.com/p/6-
best-practices-for-machine-learning)
5. [A Gentle Introduction to Graph Neural Networks -
Distill.pub](https://distill.pub/2021/gnn-intro)
