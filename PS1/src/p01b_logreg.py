import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***

    # Train the model 
    model = LogisticRegression()
    model.fit(x_train, y_train)


    # Plot the results
    util.plot(x_train, y_train, model.theta, 'Plots/p01b_{}.png'.format(pred_path[-5]))

    # Calculate predictions
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_hat = model.predict(x_eval)
    np.savetxt(pred_path, y_hat > 0.5, fmt='%d')

    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        
        # In logistic regression we will maximize the log-likelihood (l(theta))
        # instead of minimizing the cost function 

        # parameters will update according to the following rule
        # theta = theta - inv(H).dot(gradient of l(theta))

        ## Initialize theta with zeros
        m, n = x.shape
        self.theta = np.zeros(n)  # (n x 1)

        while (True):
            theta_0 = np.copy(self.theta)   # old theta
            h_theta = 1 / (1 + np.exp(-x.dot(self.theta)))   # hypothesis function (sigmoid) (m x 1)
            dh_theta = h_theta.dot((1 - h_theta).T)  # derivative of hypothesis function (m x m)
            hessian = -x.T.dot(dh_theta).dot(x)    # hessian (n x n)
            dl = ((y - h_theta).T).dot(x)    # gradient of log likelihood (1 x n)
            self.theta = self.theta - np.linalg.inv(hessian).dot(dl.T)   # parameter update (n x 1)

            if np.linalg.norm(self.theta-theta_0, ord=1) < self.eps:
                break

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        
        # hypothesis function (sigmoid)
        return 1 / (1 + np.exp(-x.dot(self.theta)))
        # *** END CODE HERE ***
