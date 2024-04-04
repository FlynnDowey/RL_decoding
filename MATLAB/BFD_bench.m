clear; close all; clc
run_bch = true;
run_rm = true;
channelType = "AWGN";

load('../Hmat/BCH_63_45_std.mat', 'H')
%% BCH
if run_bch == true
    n = 63;   % Codeword length
    k = 45;   % Message length
    nwords = 1000; % Number of words to encode
    msgTx = gf(randi([0 1],nwords,k));
    enc = bchenc(msgTx,n,k);
    trial = 1;

    if channelType =="BSC"
       noise = linspace(0.01, 0.45, 10);
    elseif channelType == "AWGN"
        noise = linspace(1, 7, 10);
        enc_reals = forwardMapping(enc.x);
    end
    
    BER = zeros(length(noise));

    for idx = 1:length(noise)
        if channelType == "BSC"
            noisycode = bsc(enc, noise(idx)); % Binary symmetric channel
        elseif channelType == "AWGN"
            noisycode_reals = awgn(enc_reals, noise(idx)); % AWGN noise
            noisycode = gf(backwardMapping(noisycode_reals));
        end
        c_tilde = noisycode.x;

        ccode = BFD(c_tilde, H);
        x = sum(xor(enc.x, ccode), 2);
        BER(trial) = sum(x, 1) / (n*length(x));
        trial = trial + 1;
    end
    
    fileName = append('../benchmark/BFD_BER_BCH_', channelType, '_63_45.mat');
    save(fileName, "BER");    
    
    figure;
    if channelType == "BSC"
        semilogy(linspace(0.45, 0.01, 10), BER, '--o')
    elseif channelType == "AWGN"
        semilogy(noise, BER, '--o')
    end
end

%% Reed-Muller 
if run_rm == true
    load("../Hmat/RM_3_6_std.mat")
    [k, n] = size(G);
    nwords = 1000; % Number of words to encode
    
    msgTx = randi([0 1],nwords,k);
    enc = mod(msgTx*G, 2);
    BER = zeros(10);
    trial = 1;

    if channelType =="BSC"
       noise = linspace(0.01, 0.45, 10);
    elseif channelType == "AWGN"
        noise = linspace(1, 7, 10);
        enc_reals = forwardMapping(enc);
    end


    for idx = 1:length(noise)
        if channelType == "BSC"
            noisycode = bsc(enc, noise(idx)); % Binary symmetric channel
        elseif channelType == "AWGN"
            noisycode_reals = awgn(enc_reals, noise(idx)); % AWGN noise
            noisycode = backwardMapping(noisycode_reals);
        end
        ccode = BFD(noisycode, H);
        x = sum(xor(enc, ccode), 2);
        BER(trial) = sum(x, 1) / (n*length(x));
        trial = trial + 1;
    end
    
    
    fileName = append('../benchmark/BFD_BER_RM_', channelType, '_3_6.mat');
    save(fileName, "BER");    
    
    figure;
    if channelType == "BSC"
        semilogy(linspace(0.45, 0.01, 10), BER, '--o')
    elseif channelType == "AWGN"
        semilogy(noise, BER, '--o')
    end
end