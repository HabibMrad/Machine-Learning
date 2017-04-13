function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
m = length(y); % number of training examples

delta = (X*theta)-y;
J = (delta'*delta + lambda * theta(2:end)'*theta(2:end))/(2*m);

grad = (X' * delta + [0; lambda*theta(2:end)])/m;

end
