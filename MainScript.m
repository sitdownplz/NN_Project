%% NN Project Main Script
% 2022/08/19 edited by Tim
% This is the main script that demostrate the NN construct and training
% process
clear,clc
close all

%% Create Dummy Data
% Create demostrate dummy data for training
n = 20;
x=linspace(-1,1,n); y=x;
[X,Y] = meshgrid(x,y);
V = X.^2+Y.^2;
figure, scatter3(X,Y,V)

predictor = [X(:), Y(:)];
response = V(:);

%% Create Network Framework

% rng(1)    % Fixed random seed

% Create model layer
inputlayer = NNLayer(2, 4);
h1 = NNLayer(3, 4);
h2 = NNLayer(4, 4);
h3 = NNLayer(4, 3);
outputlayer = NNLayer(3,1);
% outputlayer.actFcn = @tanh;

% Combine layer into model
% lgraph = [inputlayer,h1,h2,h3,outputlayer];
lgraph = [inputlayer,h3,outputlayer];
mdl = combineNNLayer(lgraph);

% Create optimizer object
optim = optimizer(mdl);
optim.batch_size = 10;

Elapsed_time = 0;
figure
h = animatedline;

data_length = length(predictor);
% Start training
for iter = 1:500    % Iteration
    total_Loss = [];
    data_index = randperm(data_length);
    if mod(iter,100)==0
        optim.lr = optim.lr*0.3;
        optim.m = optim.m*0.8;
    end
    for i  = 1:data_length    % Epoch
        tic
        idx = data_index(i);
        out = mdl.eval(predictor(idx,:), response(idx));
        optim.step(iter)
        epoch_time = toc;
        Elapsed_time = Elapsed_time + epoch_time;
        total_Loss = [total_Loss, optim.Loss];
    end
    title(sprintf('Total Training Time: %f\nMean Loss: %f\n Best Loss: %f\n',...
        Elapsed_time, mean(total_Loss)),optim.BestLoss)
    addpoints(h,iter,mean(total_Loss))
    drawnow

end

% Evaluate trained model
out = mdl.eval(predictor, response);
figure,scatter3(predictor(:,1), predictor(:,2), out)