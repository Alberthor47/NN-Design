% Transfer function for Perceptron
% Hardlim transfer function
function y = hardlim(x)
    y = double(x >= 0);
end
