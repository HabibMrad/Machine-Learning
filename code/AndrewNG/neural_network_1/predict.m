function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);

a1 = [ones(m,1) X]; %adding bias unit to X making a1

a2 = sigmoid(a1*Theta1'); %calculating a2
a2 = [ones(m, 1) a2]; %adding bias unit

a3 = sigmoid(a2*Theta2'); %calculating a3

[value, p] = max(a3,[],2); %transforming a3 into predictions based on max value



end
