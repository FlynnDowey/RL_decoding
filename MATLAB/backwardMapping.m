function X = backwardMapping(Y)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    X = zeros(size(Y));
    X(sign(Y) > 0) = 0;
    X(sign(Y) < 0) = 1;
end