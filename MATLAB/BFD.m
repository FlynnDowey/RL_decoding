function [c_return] = BFD(z, H)
[num, n] = size(z);
c_return = zeros(size(z));
for i = 1:num
    c_hat = double(z(i, :));
    iters = 1;
    max_iters = 10;
    while true
        syndrome = all(mod(H*c_hat',2) == 0);
        if ~syndrome & logical(iters <= max_iters)
            s = mod(H*c_hat', 2);
            V = sum(s);
            Q = zeros(n, 1);
            for j = 1:n
                ej = zeros(n, 1);
                ej(j) = 1;
                s = mod(H*(mod(c_hat' + ej, 2)),2);
                Q(j) = V - sum(s);
            end
            e_best = zeros(n, 1);
            [~, idx] = max(Q);
            e_best(idx) = 1;
            c_hat = mod(c_hat + e_best', 2);
            iters = iters + 1;
        else
            break;
        end
    end
    c_return(i, :) = c_hat;
end
