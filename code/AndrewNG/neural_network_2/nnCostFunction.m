function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

%FEEDFORWARD PROPAGATION

a1 = [ones(1,m); X']; %adding bias unit to X making a1
a2 = [ones(1,m); sigmoid(Theta1*a1)]; %calculating a2 + bias unit

h = sigmoid(Theta2*a2); %calculating a3


%Change y to Y 0-1 labeled
Y = zeros(num_labels, m);
Y(sub2ind(size(Y), y', 1:m)) = 1;


%CALCULATE J
%Possible to go faster?

J = - (sum((Y.*log(h) + (1-Y).*log(1-h))(:))... %Unregularized
		- lambda/2 ...
		* (sum(Theta1(:,2:end)(:).^2) + sum(Theta2(:,2:end)(:).^2))) ...
		/ m; %only one division of sum by m

%BACKPROPAGATION

%deltas
delta3 = h - Y;
delta2 = (Theta2'*delta3).*(a2.*(1-a2)); %Faster version of sigmoidGradient

%GRADIENTS
Theta2_grad = (delta3*a2' + lambda * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)])/m;
Theta1_grad = (delta2(2:end, :)*a1' + lambda * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)])/m; %regularization in delta2?

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
