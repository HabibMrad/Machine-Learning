function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha
%		returns J_history which is the computed cost of the function in each iteration
%		useful for debugging; it shows whether J drops or not

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

		theta = theta - (alpha/m) * X' * (X*theta - y);

    J_history(iter) = computeCostMulti(X, y, theta);

end

end
