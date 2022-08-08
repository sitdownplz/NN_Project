X = [3,4];
W = [0.2,0.2];
b = [0,0];

rng(1)
inputlayer = NNLayer(2, 4);
h1 = NNLayer(4, 4);
h2 = NNLayer(4,1);
outputlayer = NNLayer(1);
lgraph = [inputlayer,h1,h2,outputlayer];
lgraph = [inputlayer,outputlayer];
mdl = combineNNLayer(lgraph);

Y = mdl.eval(X)