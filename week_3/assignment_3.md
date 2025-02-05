

### Question 1:
**Which of the following statement(s) about decision boundaries and discriminant functions of classifiers is/are true?**

- **Option 1:** In a binary classification problem, all points \(x\) on the decision boundary satisfy \(\delta_1(x) = \delta_2(x)\).
  - **True:** For binary classification, the decision boundary is where the discriminant functions of the two classes are equal.

- **Option 2:** In a three-class classification problem, all points on the decision boundary satisfy \(\delta_1(x) = \delta_2(x) = \delta_3(x)\).
  - **False:** In a three-class problem, the decision boundary is where at least two discriminant functions are equal, not necessarily all three.

- **Option 3:** In a three-class classification problem, all points on the decision boundary satisfy at least one of \(\delta_1(x) = \delta_2(x)\), \(\delta_2(x) = \delta_3(x)\), or \(\delta_3(x) = \delta_1(x)\).
  - **True:** This correctly describes the decision boundaries in a three-class problem.

- **Option 4:** If \(x\) does not lie on the decision boundary then all points lying in a sufficiently small neighbourhood around \(x\) belong to the same class.
  - **True:** Points not on the decision boundary are in regions where one class dominates.

**Correct Options:** 1, 3, 4

### Question 2:
**You train an LDA classifier on a dataset with 2 classes. The decision boundary is significantly different from the one obtained by logistic regression. What could be the reason?**

- **Option 1:** The underlying data distribution is Gaussian.
  - **Not necessarily a reason for difference:** LDA assumes Gaussian distribution, but logistic regression does not require this.

- **Option 2:** The two classes have equal covariance matrices.
  - **Not a reason for difference:** Equal covariance matrices would make LDA and logistic regression more similar.

- **Option 3:** The underlying data distribution is not Gaussian.
  - **Possible reason:** If the data is not Gaussian, LDA's assumptions are violated, leading to different decision boundaries.

- **Option 4:** The two classes have unequal covariance matrices.
  - **Possible reason:** Unequal covariance matrices can lead to different decision boundaries between LDA and logistic regression.

**Correct Options:** 3, 4

### Question 3:
**Compute the likelihood of observing the data given these model parameters.**

Given:
- \(y_i\): 1, 0, 0, 1
- \(p_1(x_i)\): 0.8, 0.5, 0.2, 0.9

Likelihood is calculated as:
\[
\text{Likelihood} = p_1(x_1)^{y_1} \cdot (1 - p_1(x_1))^{1 - y_1} \cdot p_1(x_2)^{y_2} \cdot (1 - p_1(x_2))^{1 - y_2} \cdot \ldots
\]

Calculating for each point:
1. For \(y_1 = 1\): \(0.8^1 \cdot (1 - 0.8)^{1 - 1} = 0.8\)
2. For \(y_2 = 0\): \(0.5^0 \cdot (1 - 0.5)^{1 - 0} = 0.5\)
3. For \(y_3 = 0\): \(0.2^0 \cdot (1 - 0.2)^{1 - 0} = 0.8\)
4. For \(y_4 = 1\): \(0.9^1 \cdot (1 - 0.9)^{1 - 1} = 0.9\)

Multiplying these together:
\[
0.8 \times 0.5 \times 0.8 \times 0.9 = 0.288
\]

**Correct Option:** 0.288


### Question 4:
**Which of the following statement(s) about logistic regression is/are true?**

- **Option 1:** It learns a model for the probability distribution of the data points in each class.
  - **False:** Logistic regression models the probability of a binary outcome, not the distribution of data points.

- **Option 2:** The output of a linear model is transformed to the range (0, 1) by a sigmoid function.
  - **True:** The sigmoid function transforms the linear model's output to a probability between 0 and 1.

- **Option 3:** The parameters are learned by minimizing the mean-squared loss.
  - **False:** Logistic regression typically minimizes the log-loss (cross-entropy loss), not mean-squared loss.

- **Option 4:** The parameters are learned by maximizing the log-likelihood.
  - **True:** Logistic regression parameters are estimated by maximizing the log-likelihood of the observed data.

**Correct Options:** 2, 4

### Question 5:
**Consider a modified form of logistic regression given below where \( k \) is a positive constant and \( \beta_0 \) and \( \beta_1 \) are parameters.**

Given:
\[
\log \left( \frac{1 - p(x)}{kp(x)} \right) = \beta_0 + \beta_1 x
\]

We need to solve for \( p(x) \):

1. Exponentiate both sides:
\[
\frac{1 - p(x)}{kp(x)} = e^{\beta_0 + \beta_1 x}
\]

2. Rearrange to solve for \( p(x) \):
\[
1 - p(x) = kp(x) e^{\beta_0 + \beta_1 x}
\]
\[
1 = p(x) (1 + k e^{\beta_0 + \beta_1 x})
\]
\[
p(x) = \frac{1}{1 + k e^{\beta_0 + \beta_1 x}}
\]

This matches the form:
\[
p(x) = \frac{1}{ke^{\beta_0} + e^{\beta_1 x}}
\]

**Correct Option:** 
\[
\frac{e^{-\beta_1 x}}{ke^{\beta_0} + e^{-\beta_1 x}}
\]

### Question 6:
**Consider a Bayesian classifier for a 5-class classification problem. The following tables give the class-conditioned density \( f_k(x) \) for class \( k \in \{1, 2, \ldots, 5\} \) at some point \( x \) in the input space.**

We are given a Bayesian classifier for a 5-class classification problem with the following class-conditioned densities \( f_k(x) \) and prior probabilities \( \pi_k \). The posterior probability for class \( k \) is given by:
\[
P(k | x) \propto f_k(x) \cdot \pi_k
\]

Given:
\[
\begin{array}{|c|c|c|c|c|c|}
\hline
k & 1 & 2 & 3 & 4 & 5 \\
\hline
f_k(x) & 0.15 & 0.20 & 0.05 & 0.50 & 0.01 \\
\hline
\end{array}
\]

- **Option 1:** The predicted label at \( x \) will always be class 4.
   - **False**: The predicted label depends on both \( f_k(x) \) and \( \pi_k \). Without knowing \( \pi_k \), we cannot assert that class 4 will always be the predicted label. For example, if \( \pi_4 \) is very small, class 4 might not be the predicted label.

- **Option 2:** If \( 2\pi_i \leq \pi_{i+1} \forall i \in \{1, \ldots, 4\} \), the predicted class must be class 4.
   - **True**: This condition implies that the prior probabilities are increasing (\( \pi_1 \leq \pi_2 \leq \pi_3 \leq \pi_4 \leq \pi_5 \)). Given that \( f_4(x) = 0.50 \) is the highest, the posterior probability for class 4 will be the highest, making it the predicted class.

- **Option 3:** If \( \pi_i \geq \frac{3}{2}\pi_{i+1} \forall i \in \{1, \ldots, 4\} \), the predicted class must be class 1.
   - **False**: This condition implies that the prior probabilities are decreasing (\( \pi_1 \geq \frac{3}{2}\pi_2 \geq \frac{3}{2}\pi_3 \geq \frac{3}{2}\pi_4 \geq \frac{3}{2}\pi_5 \)). However, even if \( \pi_1 \) is large, \( f_1(x) = 0.15 \) is not the highest. If \( \pi_4 \) is sufficiently large, class 4 could still have a higher posterior probability than class 1.

- **Option 4:** The predicted label at \( x \) can never be class 5.
   - **False**: Although \( f_5(x) = 0.01 \) is the lowest, if \( \pi_5 \) is sufficiently large, class 5 could still be the predicted label.

**Correct Options:** 2, 


### Question 7:
**Which of the following statement(s) about a two-class LDA classification model is/are true?**

- **Option 1:** On the decision boundary, the prior probabilities corresponding to both classes must be equal.
   - **False**: The decision boundary is determined by the equality of the posterior probabilities, not necessarily the prior probabilities. The priors \( \pi_1 \) and \( \pi_2 \) can be unequal.

- **Option 2:** On the decision boundary, the posterior probabilities corresponding to both classes must be equal.
   - **True**: The decision boundary is defined where the posterior probabilities of the two classes are equal.

- **Option 3:** On the decision boundary, class-conditioned probability densities corresponding to both classes must be equal.
   - **False**: The class-conditioned densities \( f_1(x) \) and \( f_2(x) \) do not need to be equal on the decision boundary. The equality is in the product \( f_1(x) \cdot \pi_1 = f_2(x) \cdot \pi_2 \), not necessarily \( f_1(x) = f_2(x) \).


- **Option 4:** On the decision boundary, the class-conditioned probability densities corresponding to both classes may or may not be equal.
   - **True**: The class-conditioned densities \( f_1(x) \) and \( f_2(x) \) may or may not be equal on the decision boundary. The equality is in the product \( f_1(x) \cdot \pi_1 = f_2(x) \cdot \pi_2 \), which does not require \( f_1(x) = f_2(x) \).

**Correct Options:** 2, 4

### Question 8:
**Consider the following two datasets and two LDA classifier models trained respectively on these datasets.**

Given:
- **Dataset A:** 200 samples of class 0; 50 samples of class 1
- **Dataset B:** 200 samples of class 0 (same as Dataset A); 100 samples of class 1 created by repeating twice the class 1 samples from Dataset A

The decision boundary is of the form \( w^T x + b = 0 \).

- **Option 1:** The learned decision boundary will be the same for both models.
  - **False:** The decision boundary depends on the class distributions and priors, which change between datasets.

- **Option 2:** The two models will have the same slope but different intercepts.
  - **True:** The slope depends on the covariance matrices, which are the same, but the intercept depends on the class priors, which differ.

- **Option 3:** The two models will have different slopes but the same intercept.
  - **False:** The slopes are the same, but the intercepts differ.

- **Option 4:** The two models may have different slopes and different intercepts.
  - **False:** The slopes are the same, but the intercepts differ.

**Correct Option:** 2

### Question 9:
**Which of the following statement(s) about LDA is/are true?**

- **Option 1:** It minimizes the inter-class variance relative to the intra-class variance.
  - **False:** LDA maximizes the inter-class variance relative to the intra-class variance.

- **Option 2:** It maximizes the inter-class variance relative to the intra-class variance.
  - **True:** This is the goal of LDA.

- **Option 3:** Maximizing the Fisher information results in the same direction of the separating hyperplane as the one obtained by equating the posterior probabilities of classes.
  - **True:** Both approaches lead to the same decision boundary in LDA.

- **Option 4:** Maximizing the Fisher information results in a different direction of the separating hyperplane from the one obtained by equating the posterior probabilities of classes.
  - **False:** They result in the same direction.

**Correct Options:** 2, 3

### Question 10:
**Which of the following statement(s) regarding logistic regression and LDA is/are true for a binary classification problem?**

- **Option 1:** For any classification dataset, both algorithms learn the same decision boundary.
  - **False:** They learn the same decision boundary only under specific conditions (e.g., Gaussian distributions with equal covariance).

- **Option 2:** Adding a few outliers to the dataset is likely to cause a larger change in the decision boundary of LDA compared to that of logistic regression.
  - **True:** LDA is more sensitive to outliers because it relies on mean and covariance estimates.

- **Option 3:** Adding a few outliers to the dataset is likely to cause a similar change in the decision boundaries of both classifiers.
  - **False:** LDA is generally more affected by outliers.

- **Option 4:** If the intra-class distributions deviate significantly from the Gaussian distribution, logistic regression is likely to perform better than LDA.
  - **True:** Logistic regression does not assume Gaussian distributions and can perform better in such cases.

**Correct Options:** 2, 4
