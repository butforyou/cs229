import numpy as np
import util
import matplotlib.pyplot  as plt

def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    # *** START CODE HERE ***
    # Train a logistic regression classifier
    tareget_model=LogisticRegression()

    tareget_model.fit(x_train,y_train)

    
    # Plot decision boundary on top of validation set set

    y_save=tareget_model.predict(x_valid)
    plot_path_true = save_path.replace('.txt', '.png')
    util.plot(x_train,y_train,tareget_model.theta,plot_path_true)
    plt.figure(figsize=(10, 8))
    
    # 绘制数据点
    plt.scatter(x_valid[y_valid == 1, 1], x_valid[y_valid == 1, 2], 
                c='red', marker='o', label='Class 1', alpha=0.7)
    plt.scatter(x_valid[y_valid == 0, 1], x_valid[y_valid == 0, 2], 
                c='blue', marker='x', label='Class 0', alpha=0.7)
    # Use np.savetxt to save predictions on eval set to save_path
    plt.show()
    np.savetxt(save_path,y_save)
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
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
        self.theta = np.zeros(x.shape[1])
        for i in range(0,self.max_iter):

            z=x.dot(self.theta)
            sigmoid=1/(1+np.exp(-z))

            #梯度
            eps=1/len(y)*(sigmoid-y).dot(x)

            #梯度下降
            self.theta=self.theta-self.step_size*eps
            if np.linalg.norm(eps)<self.eps:
                break
          
    '''
        for i in range(self.max_iter):
         z = X @ self.theta
         h = 1 / (1 + np.exp(-z))  # sigmoid
        
        # 梯度
         grad = X.T @ (h - y)
        
        # Hessian 矩阵
         R = np.diag(h * (1 - h))
         H = X.T @ R @ X
        
        # 更新
         delta = np.linalg.inv(H) @ grad
         self.theta -= delta
        
        # 收敛条件
         if np.linalg.norm(delta) < self.eps:
            break
        # *** END CODE HERE ***
     '''
    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        sigmoid=1/(1+np.exp(-x.dot(self.theta)))
        return sigmoid
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
