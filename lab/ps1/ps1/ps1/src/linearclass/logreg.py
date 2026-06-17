import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    # Plot decision boundary on top of validation set set
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    util.plot(x_valid, y_valid, logreg.theta, save_path.replace('.txt', '.png'))
    # Use np.savetxt to save predictions on eval set to save_path
    y_pred = logreg.predict(x_valid)
    np.savetxt(save_path, y_pred)
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n, d = x.shape  # n是样本数，d是特征数
        self.theta = np.zeros(d) # 初始化参数为0

        # 设定牛顿法收敛的超参数
        max_iter = self.max_iter  # 牛顿法通常几次迭代就能收敛，不用设太多
        tol = self.eps     # 梯度模长的阈值，低于此值停止迭代
        step_size = self.step_size
        for i in range(max_iter):
            # 1. 计算当前参数下的预测概率 h (z = theta^T x, h = sigmoid(z))
            # h 的形状是 (n,)
            z = x @ self.theta
            h = 1 / (1 + np.exp(-z)) # 逻辑回归的 Sigmoid 函数
            
            # 2. 计算梯度 (Gradient)
            # 根据推导：grad = (1/n) * X^T * (h - y)
            # x.T 形状 (d, n), (h-y) 形状 (n,) -> 结果 (d,)
            grad = (1 / n) * (x.T @ (h - y))

            # 3. 检查是否收敛
            if np.linalg.norm(grad) < tol:
                print(f"Converged at iteration {i}")
                break

            # 4. 计算海森矩阵 (Hessian Matrix)
            # 根据推导：H = (1/n) * X^T * D * X
            # D 是对角矩阵，对角线元素为 h_i * (1 - h_i)
            # 在 Numpy 中，利用广播机制，构造 diag(D) 的向量，形状 (n,)
            diag_D = h * (1 - h) 
            
            # 构造海森矩阵，形状 (d, d)
            # 注意：diag_D[:, None] 将形状 (n,) 变为 (n, 1) 以实现广播乘法
            H = (1 / n) * (x.T @ (diag_D[:, None] * x))

            # 5. 更新参数 (牛顿法更新步骤)
            # 注意：不要使用 np.linalg.inv(H) @ grad (效率低且不稳定)
            # 正确的做法是求解线性方程组 H * delta = grad
            # 解出来的 delta 就是牛顿法计算的更新步长
            delta = np.linalg.solve(H, grad)
            
            # 更新 theta
            self.theta -= step_size*delta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        h = 1/(1+np.exp(-x@self.theta))
        return h
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
