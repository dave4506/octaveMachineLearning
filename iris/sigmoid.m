%% ======================= Sigmoid function =======================
function g = sigmoid(z)

g = zeros(size(z));

alfa = exp(-1*z);
bravo = 1 + alfa;
charlie = 1 ./ bravo;
g = charlie;

end
