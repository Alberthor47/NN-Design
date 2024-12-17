% RecurrentLayer for Hamming Network
classdef recurrentLayer
    properties
        Weights
        TransferFunction
    end
    
    methods
        function obj = recurrentLayer()
            % Constructor with optional weights and transfer function
            obj.Weights = [];
            obj.TransferFunction = @poslin; % Positive linear transfer function
        end
        
        function result = propagate(obj, initial_a, aes)
            % Set W based on initial data
            if isempty(obj.Weights)
                s = size(initial_a, 1);
                epsilon = 1 / (s - 1) - aes; % Adjust epsilon slightly
                W = -epsilon * ones(s);
                for i = 1:s
                    W(i, i) = 1;
                end
                obj.Weights = W;
            end
            
            % Display metadata
            disp('TF in Recurrent Layer:');
            disp(obj.TransferFunction);
            disp('W in Recurrent Layer:');
            disp(obj.Weights);

            % Propagate through RecurrentLayer
            a2 = obj.TransferFunction(obj.Weights * initial_a);
            i = 1;
            while true
                i = i + 1;
                a3 = obj.TransferFunction(obj.Weights * a2);

                fprintf('a(%d) in Recurrent Layer:\n', i);
                disp(a2);
                if isequal(a2, a3)
                    fprintf('a(%d) in Recurrent Layer:\n', i+1);
                    disp(a3);

                    result = a3;
                    break;
                else
                    a2 = a3;
                    if i == 100
                        result = [];
                        break;
                    end
                end
            end
        end
    end
end
