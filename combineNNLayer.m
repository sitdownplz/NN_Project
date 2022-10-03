function model = combineNNLayer(layer_graph)
        model = NNmodel;
        model.LayerGraph = layer_graph;
%         model.Connection= [model.LayerGraph.neuronNum];
        %         model.GradFcn = gradFcn;
end