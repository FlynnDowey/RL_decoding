function Y = forwardMapping(X)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    Y = zeros(size(X));
    Y(X==1) = -1;
    Y(X==0) = 1;
end