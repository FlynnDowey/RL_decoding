clear; close all; clc;

run_bch = false;
run_rm = true;

%% BCH
if run_bch == true
    n = 63;   % Codeword length
    k = 45;   % Message length
    nwords = 1000; % Number of words to encode
    msgTx = gf(randi([0 1],nwords,k));
    enc = bchenc(msgTx,n,k);
    BER = zeros(10);
    trial = 1;
    for noise_i = linspace(0.01, 0.45, 10)
        noisycode = bsc(enc,noise_i); % Binary symmetric channel
        [decoded,cnumerr,ccode]= bchdec(noisycode,n,k);
        x = sum(xor(enc.x, ccode.x), 2);
        BER(trial) = sum(x, 1) / (n*length(x));
        trial = trial + 1;
    end
    
    save('../benchmark/BER_BCH_63_45.mat', "BER");
    
    
    figure;
    semilogy(linspace(0.45, 0.01, 10), BER, '--o')
end
%% Reed-Muller 
if run_rm == true
    clear; clc;
    load("../Hmat/RM_3_6_std.mat")
    [k, n] = size(G);
    nwords = 1000; % Number of words to encode
    
    msgTx = randi([0 1],nwords,k);
    enc = mod(msgTx*G, 2);
    BER = zeros(10);
    trial = 1;
    for noise_i = linspace(0.01, 0.45, 10)
        noisycode = bsc(enc,noise_i); % Binary symmetric channel
        x = 0;
        parfor code_i = 1:nwords
            [decoded_cw, ~]= rmdec_reed(noisycode(code_i, :),3,6);
            x = x + sum(xor(enc(code_i, :), decoded_cw), 2);
        end
        BER(trial) = sum(x, 1) / (n* length(x));
        trial = trial + 1;
    end
    
    save('../benchmark/BER_RM_3_6.mat', "BER");
    
    
    figure;
    semilogy(linspace(0.45, 0.01, 10), BER, '--o')
end