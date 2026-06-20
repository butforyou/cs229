import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    model = GDA()
    model.fit(x_train, y_train)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=False)
    print(f"theta: {model.theta}")
    pred = model.predict(x_valid)
    np.savetxt(save_path, pred)
    # Train a GDA classifier
    # Plot decision boundary on validation set
    util.plot(x_valid, y_valid, model.theta, save_path.replace('.txt', '.png'))
    # Use np.savetxt to save outputs from validation set to save_path
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
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
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n, d = x.shape  # n是样本数，d是特征数
        self.phi = np.mean(y)
        self.mu_0 = np.mean(x[y==0],axis = 0)
        self.mu_1 = np.mean(x[y==1],axis = 0)
        sigma = 0
        for i in range(n):
          xi = x[i]
          if y[i] == 0:
            diff = xi - self.mu_0
          else:
           diff = xi - self.mu_1
          sigma += np.outer(diff, diff)

        self.sigma = sigma / n
        # Find phi, mu_0, mu_1, and sigma
        inv_sigma = np.linalg.inv(self.sigma)
        
        # 方法1：使用索引赋值（需要确保维度匹配）
        self.theta = np.zeros(d+1)
        self.theta[0] = -0.5 * (self.mu_1 @ inv_sigma @ self.mu_1 - 
                                self.mu_0 @ inv_sigma @ self.mu_0) + np.log(self.phi / (1 - self.phi))
        self.theta[1:] = inv_sigma @ (self.mu_1 - self.mu_0)
        # Write theta in terms of the parameters
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        p = self.phi * np.exp(-0.5 * np.sum((x - self.mu_1) @ np.linalg.inv(self.sigma) * (x - self.mu_1), axis=1)) / \
            ((1 - self.phi) * np.exp(-0.5 * np.sum((x - self.mu_0) @ np.linalg.inv(self.sigma) * (x - self.mu_0), axis=1)) + self.phi * np.exp(-0.5 * np.sum((x - self.mu_1) @ np.linalg.inv(self.sigma) * (x - self.mu_1), axis=1)))
        return p
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
