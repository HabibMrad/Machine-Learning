function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

h_theta = sigmoid(X*theta);

J = ((sum(-y.*log(h_theta) - (1-y).*log(1-h_theta))) + (lambda/2)*sum(theta(2:end).^2))/m;
grad = (X' * (h_theta - y) + [0; lambda*theta(2:end)])/m;

end
