import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    clf = GDA()
    clf.fit(x_train, y_train)

    # Plot the classifier with data
    util.plot(x_train, y_train, clf.theta, 'Plots/p01e_{}.png'.format(pred_path[-5]))

    # validation dataset
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept = False)
    y_hat = clf.predict(x_eval)
    np.savetxt(pred_path, y_hat > 0.5, fmt='%d')
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        
        m, n = x.shape
        self.theta = np.zeros(n+1)
        # Calculate the parameters value, no iterations required since GDA
        y_1 = np.sum(np.where(y == 1))
        mu_0 = np.sum(x[y == 0]) / (m - y_1)
        phi = y_1 / m
        mu_1 = np.sum(x[y == 1]) /  y_1
        cov = ((x[y == 0] - mu_0).T.dot(x[y==0] - mu_0)) + (x[y == 1] - mu_1).T.dot(x[y == 1] - mu_1)

        # Computing theta 
        inv_cov = np.linalg.inv(cov)
        self.theta[0] = 0.5 * (mu_0 + mu_1).dot(inv_cov).dot(mu_0 - mu_1) - np.log((1 - phi) / phi)
        self.theta[1:] = inv_cov.dot(mu_1 - mu_0)

        return self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-x.dot(self.theta)))
        # *** END CODE HERE
