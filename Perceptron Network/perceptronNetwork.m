% Implementation of Perceptron Network (With learining)
classdef perceptronNetwork
    properties
        Weights
        Bias
        TransferFunction
    end
    
    methods
        function obj = perceptronNetwork(number_of_neurons, input_size)
            % Constructor
            obj.Weights = rand(number_of_neurons, input_size); % Build the Weight matrix, init with random values
            obj.Bias = rand(number_of_neurons, 1); % Build the Bias vector, init with random values
            obj.TransferFunction = @(x) hardlim(x); % TF in perceptron should always be hardlim
        end
        
        function output = classify(obj, prototype)
            % Classify input using current weights and bias
            net_input = obj.Weights * prototype + obj.Bias;
            output = obj.TransferFunction(net_input);
        end
        
        function obj = train(obj, prototypes, max_epochs)
            % Train perceptron using prototypes with a maximum epoch counter
            epoch = 0; % Initialize epoch counter
            while ~obj.correct(prototypes) && epoch < max_epochs
                epoch = epoch + 1; % Increment epoch counter
                for i = 1:length(prototypes)
                    input_v = prototypes{i}{1};
                    target = prototypes{i}{2};
                    
                    classification = obj.classify(input_v);
                    
                    % Update weights and bias
                    obj.Weights = obj.Weights + (target - classification) * input_v';
                    obj.Bias = obj.Bias + (target - classification);
                end
                % Log progress
                if mod(epoch, 10) == 0
                    fprintf('Epoch %d completed.\n', epoch); % Log progress
                end
            end
            
            % Check if training stopped due to reaching max epochs
            if epoch == max_epochs
                fprintf('Training stopped: maximum epochs reached (%d).\n', max_epochs);
            else
                fprintf('Training completed successfully in %d epochs.\n', epoch);
            end
        end
        
        function is_correct = correct(obj, prototypes)
            % Check if all classifications are correct
            is_correct = true;
            for i = 1:length(prototypes)
                input_v = prototypes{i}{1};
                target = prototypes{i}{2};
                
                if target ~= obj.classify(input_v)
                    is_correct = false;
                    return;
                end
            end
        end
    end
end
