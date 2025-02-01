# WEEK 2 : Assignments

### 1) **In a linear regression model $ y = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_p x_p $, what is the purpose of adding an intercept term $ (\theta_0) $?**

   - **Correct Answer:** To adjust for the baseline level of the dependent variable when all predictors are zero.
   - **Explanation:** The intercept term $ \theta_0 $ represents the expected value of the dependent variable $ y $ when all independent variables $ x_1, x_2, \ldots, x_p $ are zero. It adjusts the model to account for the baseline level of $ y $.

### 2) **Which of the following is true about the cost function (objective function) used in linear regression?**

   - **Correct Answer:** It measures the sum of squared differences between predicted and actual values.
   - **Explanation:** The cost function in linear regression typically used is the Mean Squared Error (MSE), which is the average of the squared differences between the predicted values and the actual values. This function is convex and is minimized to find the best-fitting line.

### 3) **Which of these would most likely indicate that Lasso regression is a better choice than Ridge regression?**

   - **Correct Answer:** Only a few features are truly relevant.
   - **Explanation:** Lasso regression (L1 regularization) tends to produce sparse models by shrinking some coefficients to zero, effectively performing feature selection. This makes it a better choice when only a few features are relevant.

### 4) **Which of the following conditions must hold for the least squares estimator in linear regression to be unbiased?**

   - **Correct Answer:** The errors must have a mean of zero.
   - **Explanation:** For the least squares estimator to be unbiased, the errors (residuals) must have a mean of zero. This ensures that the model is correctly specified and that the estimates are not systematically over- or under-estimating the true values.


### 5) **When performing linear regression, which of the following is most likely to cause overfitting?**

   - **Correct Answer:** Including irrelevant predictors in the model.
   - **Explanation:** Overfitting occurs when a model learns the noise in the training data rather than the underlying pattern. Including irrelevant predictors can lead to a model that is too complex and captures noise, leading to poor generalization on new data.

### 6) **You have trained a complex regression model on a dataset. To reduce its complexity, you decide to apply Ridge regression, using a regularization parameter $\lambda$. How does the relationship between bias and variance change as $\lambda$ becomes very large? Select the correct option**

   - **Correct Answer:** bias is high, variance is low.
   - **Explanation:** As $\lambda$ increases, the regularization effect becomes stronger, shrinking the coefficients more aggressively. This increases bias because the model becomes less flexible and may underfit the data. However, it reduces variance because the model is less sensitive to fluctuations in the training data.

### 7) **Given a training data set of 10,000 instances, with each input instance having 12 dimensions and each output instance having 3 dimensions, the dimensions of the design matrix used in applying linear regression to this data is**

   - **Correct Answer:** $10000 \times 13$
   - **Explanation:** The design matrix $X$ includes the input features augmented with a column of ones for the intercept term. Since each input instance has 12 dimensions, the design matrix will have $10000$ rows (instances) and $13$ columns (12 features + 1 intercept).

### 8) **The linear regression model $y = a_0 + a_1x_1 + a_2x_2 + \ldots + a_px_p$ is to be fitted to a set of $N$ training data points having P attributes each. Let $X$ be $N \times (p+1)$ vectors of input values (augmented by 1's), $Y$ be $N \times 1$ vector of target values, and $\theta$ be $(p+1) \times 1$ vector of parameter values $(a_0, a_1, a_2, \ldots, a_p)$. If the sum squared error is minimized for obtaining the optimal regression model, which of the following equation holds?**

   - **Correct Answer:** $X^T X \theta = X^TY$
   - **Explanation:** The normal equations for linear regression are derived by minimizing the sum of squared errors. The solution to these equations is given by $X^T X \theta = X^TY$, which provides the optimal parameter values $\theta$.


### 9) **Which of the following scenarios is most appropriate for using Partial Least Squares (PLS) regression instead of ordinary least squares (OLS)?**

   - **Correct Answer:** When there is significant multicollinearity among predictors or the number of predictors exceeds the number of samples.
   - **Explanation:** PLS regression is particularly useful when dealing with multicollinearity (high correlation among predictors) or when the number of predictors is larger than the number of samples. PLS reduces the dimensionality of the predictors by projecting them into a lower-dimensional space, which helps in such scenarios.

### 10) **Consider forward selection, backward selection, and best subset selection with respect to the same data set. Which of the following is true?**

   - **Correct Answer:** Best subset selection can be computationally more expensive than forward selection.
   - **Explanation:** Best subset selection evaluates all possible combinations of predictors, which can be computationally very expensive, especially with a large number of predictors. Forward selection, on the other hand, starts with no predictors and adds them one by one, which is generally less computationally intensive. Backward selection starts with all predictors and removes them one by one, which can also be computationally expensive but not as much as best subset selection.
