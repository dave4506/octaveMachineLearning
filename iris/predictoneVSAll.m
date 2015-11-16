function p = predictoneVSAll(all_theta, X)

m = size(X, 1);
num_labels = size(all_theta, 1);

p = zeros(size(X, 1), 1);

alfa = sigmoid(X*all_theta');
[beta,charlie] = max(alfa,[],2);
p = charlie;
p = p .- 1;






end
