classdef optimizer <handle
    % Optimizer object required NNmodel for input
    properties
        NNmodel    % NNmodel object
        lr mustBeNumeric =1e-3    % Learning Rate
        m mustBeNumeric =0.8    % Momentum
        GradFcn = @Roy_GD
        Loss    mustBeNumeric
        BestLoss = inf
        current_iter = 0    % Current iteration
        grad_W    % Current cumulative weight gradient of each layer
        grad_b    % Current cumulative bias gradient of each layer

    end

    methods
        % Self funciton
        function self = optimizer(NNmodel)
            switch nargin
                case 1
                    self.NNmodel = NNmodel;
                    self.grad_W = cell(1, length(self.NNmodel.LayerGraph));
                    self.grad_b = cell(1, length(self.NNmodel.LayerGraph));
            end
        end
        
        % Calculate gradient and update model 
        function step(self, init)
            % step(self, current iteration)
            if init ~= self.current_iter
                self.grad_W = cell(1, length(self.NNmodel.LayerGraph));
                self.grad_b = cell(1, length(self.NNmodel.LayerGraph));
            end
            self.Loss = CostFunc.SSELoss(self.NNmodel.Y, self.NNmodel.YPred);
            if self.Loss < self.BestLoss | randi([0,1],1) %TODO
                for layer = 1:length(self.NNmodel.LayerGraph)
                    [d_W, d_b ]= self.Roy_GD(layer);
                    error = self.NNmodel.Y - self.NNmodel.YPred; 
                    gradient_W = -d_W*error;
                    gradient_b = -d_b*error;

                    if init==1
                        step_W=-self.lr*gradient_W;
                        step_b=-self.lr*gradient_b;
                    else
                        step_W= self.m*self.NNmodel.grad_W{layer}-self.lr*gradient_W;
                        step_b= self.m*self.NNmodel.grad_b{layer}-self.lr*gradient_b;
                    end

                    if isempty(self.grad_W{layer}) & isempty(self.grad_b{layer})
                        self.grad_W{layer} = gradient_W;
                        self.grad_b{layer} = gradient_b;
                    else
                        self.grad_W{layer} = self.grad_W{layer}+gradient_W;
                        self.grad_b{layer} = self.grad_b{layer}+gradient_b;
                    end

                    if init ~= self.current_iter
                        self.NNmodel.LayerGraph(layer).W  = self.NNmodel.LayerGraph(layer).W-step_W;
                        if layer <length(self.NNmodel.LayerGraph)
                            self.NNmodel.LayerGraph(layer).b  = self.NNmodel.LayerGraph(layer).b-step_b;
                        end
                        self.current_iter = init;
                    end
                end
                self.NNmodel.grad_W = self.grad_W;
                self.NNmodel.grad_b = self.grad_b;
                self.BestLoss = self.Loss;
            end
        end


        % Roy's gradient decent methods
        function [d_W, d_b]= Roy_GD(self, layer)
            seg = 51;
            pts = 1:seg;
            r = 1e-3;
            i=complex(0,1);
            beta = r*exp(i*2*pi*pts/(seg-1));
            Layer = self.NNmodel.LayerGraph(layer);
            X = Layer.X;
            h = Layer.h;
            X = repelem(X,numel(h));

            y_hat_W = X*beta + repmat(h, Layer.neuronNum,1);
            y_hat_b =h-beta;
            d_W = combCal(y_hat_W, 'weight');
            d_b = combCal(y_hat_b, 'bias');

            function d_real = combCal(y_hat, type)
                comb = [];
                if numel(h)==1
                    comb = y_hat;
                    new = self.NNmodel.backward(comb, layer);
                else
                    choose_tbl = eye(numel(h));

                    switch type
                        case 'weight'
                            choose_tbl = repmat(choose_tbl, [1, Layer.neuronNum]);
                        case 'bias'
                    end

                    for idx = 1:size(choose_tbl,2)
                        select = choose_tbl(:,idx);
                        comb = [comb, h.*~select + y_hat(idx,:).*select];
                    end

                    if layer < length(self.NNmodel.LayerGraph)
                        comb = tanh(comb);
                    end

                    new = self.NNmodel.backward(comb, layer);
                    new = reshape(new, seg, [])';
                end

                fv =  new./beta;
                weight = ones(seg,1);
                weight(2:end-1) = weight(2:end-1)*2;    % 1,2...2,1
                weight=pi*weight/(seg-1);
                d = fv*weight/(2*pi);
                d_real = real(d);
                switch type
                    case 'weight'
                        d_real = reshape(d_real,[],Layer.neuronNum);
                    case 'bias'
                end
                d_imag = imag(d);
                huge_error = abs(d_imag)>1e-8;
                if any(huge_error)
                    fprintf('Imaginary error: %e\n',d_imag(huge_error))
                end
            end
        end

    end
end