# 📚 Week 3 : Assignments  

---

## ✨ Question 1: Decision Boundaries and Discriminant Functions

### **Which of the following statement(s) about decision boundaries and discriminant functions of classifiers is/are true?**

✅ **In a binary classification problem, all points $ x $ on the decision boundary satisfy $\delta_1(x) = \delta_2(x)$.**  
➡️ *Explanation:* The decision boundary in binary classification is where the discriminant functions of two classes are equal.

❌ **In a three-class classification problem, all points on the decision boundary satisfy $\delta_1(x) = \delta_2(x) = \delta_3(x)$.**  
➡️ *Explanation:* Only two discriminant functions need to be equal, not necessarily all three.

✅ **In a three-class classification problem, all points on the decision boundary satisfy at least one of $\delta_1(x) = \delta_2(x)$, $\delta_2(x) = \delta_3(x)$.**  
➡️ *Explanation:* The decision boundary occurs where at least two discriminant functions are equal.

✅ **If $ x $ does not lie on the decision boundary, then all points lying in a sufficiently small neighborhood around $ x $ belong to the same class.**  
➡️ *Explanation:* If $ x $ is not on the boundary, it is in a region dominated by one class.

---

## 🔎 Question 2: LDA vs Logistic Regression

### **Why does the LDA classifier's decision boundary differ significantly from logistic regression?**

❌ **The underlying data distribution is Gaussian.**  
✅ **The underlying data distribution is not Gaussian.**  
✅ **The two classes have unequal covariance matrices.**  
❌ **The two classes have equal covariance matrices.**  

➡️ *Explanation:* LDA assumes Gaussian distribution with equal covariance. If the data is not Gaussian or has unequal covariances, the boundary differs from logistic regression.

---

## 🎯 Question 3: Likelihood Calculation

### **Compute the likelihood of observing the data given these model parameters:**

| $ y_i $ | 1 | 0 | 0 | 1 |
|---|---|---|---|---|
| $ p_1(x_i) $ | 0.8 | 0.5 | 0.2 | 0.9 |

**Formula:**
$
\text{Likelihood} = p_1(x_1) \times (1 - p_1(x_2)) \times (1 - p_1(x_3)) \times p_1(x_4)
$

**Calculation:**
$
0.8 \times 0.5 \times 0.8 \times 0.9 = 0.288
$

✅ **Correct Answer: 0.288**  
➡️ *Explanation:* The likelihood is the product of the probabilities for observing the given labels under the model.

---

## 📈 Question 4: Logistic Regression Properties

✅ **The output of a linear model is transformed to the range (0,1) by a sigmoid function.**  
✅ **The parameters are learned by maximizing the log-likelihood.**  
❌ **It learns a model for the probability distribution of the data points in each class.**  
❌ **The parameters are learned by minimizing the mean-squared loss.**  

➡️ *Explanation:* Logistic regression models class probabilities using a sigmoid function and optimizes parameters through log-likelihood maximization.

---

## 🏆 Question 6: Bayesian Classifier for Multi-Class Problems

✅ **If $ 2\pi_1 \leq \pi_{i+1} \forall i $, the predicted class must be class 4.**  
✅ **If $ \pi_1 \geq \frac{3}{2}\pi_{i+1} \forall i $, the predicted class must be class 1.**  
❌ **The predicted label at $ x $ will always be class 4.**  
❌ **The predicted label at $ x $ can never be class 5.**  

➡️ *Explanation:* The class with the highest posterior probability is predicted, influenced by priors and likelihood.

---

## 🎯 Question 7: Two-Class LDA

✅ **On the decision boundary, the posterior probabilities corresponding to both classes must be equal.**  
✅ **On the decision boundary, class-conditioned probability densities corresponding to both classes must be equal.**  
❌ **On the decision boundary, the prior probabilities corresponding to both classes must be equal.**  
❌ **On the decision boundary, the class-conditioned probability densities corresponding to both classes may or may not be equal.**  

➡️ *Explanation:* The decision boundary is where posterior probabilities and class-conditioned densities are equal.

---

## 🔍 Question 8: LDA Decision Boundaries

✅ **The two models may have different slopes and different intercepts.**  
❌ **The learned decision boundary will be the same for both models.**  
❌ **The two models will have the same slope but different intercepts.**  
❌ **The two models will have different slopes but the same intercept.**  

➡️ *Explanation:* The decision boundary depends on the class distributions, which vary across datasets.

---

## 📊 Question 9: LDA Optimization

✅ **It minimizes the inter-class variance relative to the intra-class variance.**  
✅ **Maximizing the Fisher information results in the same direction of the separating hyperplane as equating posterior probabilities.**  
❌ **It maximizes the inter-class variance relative to the intra-class variance.**  
❌ **Maximizing the Fisher information results in a different direction for the separating hyperplane.**  

➡️ *Explanation:* LDA maximizes inter-class separation while minimizing intra-class scatter.

---

## 🔥 Question 10: Logistic Regression vs LDA

✅ **Adding outliers is likely to change LDA’s decision boundary more than logistic regression’s.**  
✅ **If intra-class distributions deviate from Gaussian, logistic regression is likely to perform better than LDA.**  
❌ **For any dataset, both algorithms learn the same decision boundary.**  
❌ **Adding a few outliers affects both classifiers' decision boundaries similarly.**  

➡️ *Explanation:* LDA is sensitive to outliers due to its Gaussian assumption, while logistic regression does not assume Gaussianity.

---

### 🎯 *End of Assignment*  
*Well done! Keep practicing for a deeper understanding of machine learning concepts!* 🚀