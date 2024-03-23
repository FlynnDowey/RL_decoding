clear; close all; clc;

%% BCH parameters
n = 63;   % Codeword length
k = 45;   % Message length
nwords = 1000; % Number of words to encode

%% Channel
msgTx = gf(randi([0 1],nwords,k));
enc = bchenc(msgTx,n,k);
BER = zeros(10);
trial = 1;
for noise_i = linspace(0.01, 0.45, 10)
    noisycode = bsc(enc,noise_i); % Binary symmetric channel
    [decoded,cnumerr,ccode]= bchdec(noisycode,n,k);
    x = sum(xor(enc.x, ccode.x), 2);
    BER(trial) = sum(x, 1) / length(x);
    trial = trial + 1;
end

save('./benchmark/BER_BCH_63_45.mat', "BER");


figure;
semilogy(linspace(0.45, 0.01, 10), BER, '--o')

%% Cyclic
clear; clc;
n = 63;
k = 51;
pol = cyclpoly(n,k);
[H,G,k] = cyclgen(n,pol);
msgTx = randi([0 1],n,k);
enc = encode(msgTx,n,k,'linear/binary',G);
BER = zeros(10);
trial = 1;
for noise_i = linspace(0.01, 0.45, 10)
    noisycode = bsc(enc,noise_i); % Binary symmetric channel
    [msg,err,ccode,cerr]= decode(noisycode,n,k,'linear/binary',G);
    x = sum(xor(enc, ccode), 2);
    BER(trial) = sum(x, 1) / length(x);
    trial = trial + 1;
end

save('./benchmark/BER_Cyclic_63_45.mat', "BER");


figure;
semilogy(linspace(0.45, 0.01, 10), BER, '--o')