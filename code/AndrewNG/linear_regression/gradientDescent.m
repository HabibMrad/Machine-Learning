function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha
%		returns J_history which is the computed cost with given theta
%		useful for debugging, as you can see whether J is dropping or not


m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

		theta = theta - (alpha/m) * X' * (X*theta - y);


    J_history(iter) = computeCost(X, y, theta);

end

end
