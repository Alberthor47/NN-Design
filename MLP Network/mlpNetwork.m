% Implementation of MLP (Multi-Layer Perceptron)
classdef mlpNetwork
    properties
        Layers                  % Number of neurons in each layer
        Weights                 % Cell array of weight matrices
        Biases                  % Cell array of bias vectors
        TransferFuncs           % Cell array of activation functions
        TransferFuncsDeriv      % Cell array of derivative functions
        TrainingErrorHistory    % Matrix of training error history
        ValidationErrorHistory  % Matrix of validation error history
    end
    
    methods
        function obj = mlpNetwork(layers, transfer_funcs)
            % Constructor: Initialize the MLP structure
            obj.Layers = layers;

            % Validate that the number of transfer functions matches the number of layers - 1
            if length(transfer_funcs) ~= length(layers) - 1
                error('The number of transfer functions must match the number of layers minus 1.');
            end
            
            % Set up transfer functions & derivatives
            obj.TransferFuncs = cell(1, length(transfer_funcs));
            obj.TransferFuncsDeriv = cell(1, length(transfer_funcs)); 
            for i = 1:length(transfer_funcs)
                switch transfer_funcs(i)
                    case 1
                        obj.TransferFuncs{i} = @(x) purelin(x);
                        obj.TransferFuncsDeriv{i} = @(x) purelind(x);
                    case 2
                        obj.TransferFuncs{i} = @(x) logsig(x);
                        obj.TransferFuncsDeriv{i} = @(x) logsigd(x);
                    case 3
                        obj.TransferFuncs{i} = @(x) tansig(x);
                        obj.TransferFuncsDeriv{i} = @(x) tansigd(x);
                    otherwise
                        error('Invalid transfer function. Use 1 (purelin), 2 (logsig), or 3 (tansig).');
                end
            end
            
            % Initialize weights and biases
            obj.Weights = cell(length(layers) - 1, 1);
            obj.Biases = cell(length(layers) - 1, 1);
            for i = 1:length(layers) - 1
                obj.Weights{i} = -1 + 2 * rand(layers(i+1), layers(i));
                obj.Biases{i} = -1 + 2 * rand(layers(i+1), 1);
            end

            % Initialize erros
            obj.TrainingErrorHistory = []; % History of training errors
            obj.ValidationErrorHistory = []; % History of validation errors
        end
        
        function obj = setWeightsAndBiases(obj, weights, biases)
            % Validate that the dimensions match
            if length(weights) ~= length(obj.Weights)
                error('The number of weight matrices does not coincide with the number of layers.');
            end
            
            if length(biases) ~= length(obj.Biases)
                error('The number of bias vectors does not coincide with the number of layers.');
            end
        
            % Validate the dimensions of each weight matrix and bias vector.
            for i = 1:length(weights)
                if ~isequal(size(weights{i}), size(obj.Weights{i}))
                    error(['The dimensions of the Weight matrix in layer ' num2str(i) ' do not match.']);
                end
                if ~isequal(size(biases{i}), size(obj.Biases{i}))
                    error(['The dimensions of the Bias matrix in layer ' num2str(i) ' do not match.']);
                end
            end
            
            % Set values
            obj.Weights = weights;
            obj.Biases = biases;
        end

        function obj = setErrorHistory(obj, training_error_history, validation_error_history)
            % Set values
            obj.TrainingErrorHistory = training_error_history;
            obj.ValidationErrorHistory = validation_error_history;
        end

        function [obj, stop_code] = train(obj, train_set, val_set, learning_rate, max_epochs, error_train, epoch_val, num_val)
            num_val_increments = 0; % Counter for consecutive increases in validation error
            val_error_prev = inf;  % Previous validation error for comparison
            stop_code = 0; % If training is completed successfully stop code is 1 else 0
            epoch = 1;
        
            while epoch <= max_epochs
                % Validation Check
                if mod(epoch, epoch_val) == 0
                    % Validation epoch
                    val_error = 0;
                    for j = 1:size(val_set, 1) % Loop over validation samples
                        input = val_set(j, 1);    % Input (single value)
                        target = val_set(j, 2);   % Target (single value)
        
                        % Forward pass
                        [val_output, ~] = obj.forward(input);

                        % Compute error
                        sample_error = target - val_output;

                        % Accumulate mean squared error for the sample
                        val_error = val_error + sample_error.^2;
                    end
                    
                    % Compute mean validation error
                    val_error = val_error / size(val_set, 1);
                    obj.ValidationErrorHistory = [obj.ValidationErrorHistory; val_error];
        
                    % Check for consecutive increases in validation error
                    if val_error > val_error_prev
                        num_val_increments = num_val_increments + 1;
                    else
                        num_val_increments = 0;
                    end
                    val_error_prev = val_error;
        
                    % Early stopping
                    if num_val_increments >= num_val
                        stop_code = 1;
                        disp('Early stopping triggered: Validation error increased consecutively.');
                        break;
                    end
                else 
                    % Training Epoch 
                    epoch_error = 0; % Accumulate training error over all samples
                    for i = 1:size(train_set, 1) % Loop over training samples
                        input = train_set(i, 1);    % Input (single value)
                        target = train_set(i, 2);   % Target (single value)
            
                        % Forward pass
                        [output, activations] = obj.forward(input);
    
                        % Compute error
                        sample_error = target - output;
            
                        % Backpropagation
                        gradients = obj.backward(activations, output, target, input);
            
                        % Update weights and biases
                        obj = obj.update_parameters(gradients, learning_rate);
            
                        % Accumulate mean squared error for the sample
                        epoch_error = epoch_error + sample_error^2;
                    end

                    % Calculate mean training error for the epoch
                    train_error = epoch_error / size(train_set, 1);
                    obj.TrainingErrorHistory = [obj.TrainingErrorHistory; train_error];
                   
                    % Stop if training error is sufficiently low
                    if train_error <= error_train
                        stop_code = 1;
                        disp('Training error goal reached.');
                        break;
                    end
                end

                epoch = epoch + 1;
            end
        end

        function [output, activations] = forward(obj, input)
            % Forward pass: Compute tf for all layers
            activations = cell(length(obj.Weights), 1);
            output = input;

            for i = 1:length(obj.Weights)
                z = obj.Weights{i} * output + obj.Biases{i};
                output = obj.TransferFuncs{min(i, length(obj.TransferFuncs))}(z); % Apply tf
                activations{i} = output;
            end
        end

        function gradients = backward(obj, activations, output, target, input)
            % Backpropagation: Compute gradients for weights and biases
            num_layers = length(obj.Weights);
            
            % Initialize gradients for all layers
            gradients.dWeights = cell(num_layers, 1);
            gradients.dBiases = cell(num_layers, 1);

            % Compute output layer s
            % s^M = -2 * F'^M(n^M) * (t - a)
            s = -2 * obj.TransferFuncsDeriv{end}(output) .* (target - output);
            
            % Backpropagate through hidden layers
            for m = num_layers:-1:1
                % Previous layer activations (or input if first layer)
                if m == 1
                    prev_activation = input;
                else
                    prev_activation = activations{m - 1};
                end

                % Compute weight and bias gradients
                gradients.dWeights{m} = s * prev_activation';
                gradients.dBiases{m} = s;
                
                % Propagate error backward if not input layer
                if m > 1
                    % s^m = F'^m(n^m) * (W^{m+1})^T * s^{m+1}
                    activation_deriv = obj.TransferFuncsDeriv{m - 1}(activations{m - 1});
                    s = activation_deriv .* (obj.Weights{m}' * s);
                end
            end
        end

        function obj = update_parameters(obj, gradients, learning_rate)
            % Update weights and biases using gradients
            for i = 1:length(obj.Weights)
                obj.Weights{i} = obj.Weights{i} - learning_rate * gradients.dWeights{i};
                obj.Biases{i} = obj.Biases{i} - learning_rate * gradients.dBiases{i};
            end
        end

        function graphErroHistory(obj, epoch_val)
            % Gráfica de evolución del error de aprendizaje y validación
            figure;
            hold on;
        
            % Graficar errores de aprendizaje como puntos verdes
            plot(1:length(obj.TrainingErrorHistory), obj.TrainingErrorHistory, 'g.-', ...
                'MarkerSize', 10, 'DisplayName', 'Error de Aprendizaje');
        
            % Graficar errores de validación como diamantes rojos
            if ~isempty(obj.ValidationErrorHistory)
                val_epochs = epoch_val * (1:length(obj.ValidationErrorHistory));
                plot(val_epochs, obj.ValidationErrorHistory, 'rd-', ...
                    'MarkerSize', 10, 'DisplayName', 'Error de Validación');
            end
        
            % Configurar título, etiquetas y leyenda
            title('Evolución del Error de Aprendizaje y Validación');
            xlabel('Época');
            ylabel('Error Cuadrático Medio (MSE)');
            legend('show');
            grid on;
            hold off;
        end

        function graphTestSet(obj, test_set)
            % Inicializar vectores
            inputs = test_set(:, 1);           % Entradas del conjunto de prueba
            target_outputs = test_set(:, 2);   % Salidas deseadas
            mlp_outputs = zeros(size(inputs)); % Salidas del MLP
        
            % Generar salidas del MLP para cada entrada del conjunto de prueba
            for i = 1:length(inputs)
                [output, ~] = obj.forward(inputs(i));
                mlp_outputs(i) = output;
            end
        
            % Crear la gráfica
            figure;
            hold on;
            
            % Graficar los valores del conjunto de prueba (círculos sin relleno)
            plot(inputs, target_outputs, 'o', 'MarkerSize', 8, 'LineWidth', 1.5, ...
                'DisplayName', 'Conjunto de Prueba');
            
            % Graficar las salidas del MLP (cruces)
            plot(inputs, mlp_outputs, 'x', 'MarkerSize', 8, 'LineWidth', 1.5, ...
                'DisplayName', 'Salidas del MLP');
        
            % Configurar la gráfica
            title('Comparación del Conjunto de Prueba vs Salidas del MLP');
            xlabel('Entrada');
            ylabel('Salida');
            legend('show');
            grid on;
            hold off;
        end

        function [trainSet, valSet, testSet] = generateSets(obj, data)
            % Take the value column
            data = data(:,2);

            % Ensure column format
            data = data(:);

            % Calculate the number of samples
            numSamples = length(data);

            % Generate random shuffled indices
            %rng('default'); % For reproducibility
            indices = randperm(numSamples);

            % Split percentages
            numTrain = floor(0.8 * numSamples); % Training 80%
            numVal = floor(0.1 * numSamples);   % Validation 10%

            % Split the data
            trainIdx = indices(1:numTrain);
            valIdx = indices(numTrain + 1:numTrain + numVal);
            testIdx = indices(numTrain + numVal + 1:end);

            % Plot the data
            figure;
            hold on;
            plot(data, 'b', 'DisplayName', 'Original Data');
            scatter(trainIdx, data(trainIdx), 10, 'g', 'DisplayName', 'Training Data');
            scatter(valIdx, data(valIdx), 10, 'o', 'DisplayName', 'Validation Data');
            scatter(testIdx, data(testIdx), 10, 'r', 'DisplayName', 'Test Data');
            
            % Add labels and legend
            title('Data Split Visualization');
            xlabel('Index');
            ylabel('Value');
            legend('show');
            grid on;
            hold off;

            % Create subsets
            trainSet = [trainIdx(:) data(trainIdx)];
            valSet = [valIdx(:) data(valIdx)];
            testSet = [testIdx(:) data(testIdx)];
        end
    end
end

% Transfer function for MLP
function y = purelin(x)
    y = x; % Linear
end

function y = purelind(x)
    y = 1; % Linear derivative
end

function y = logsig(x)
    y = 1 ./ (1 + exp(-x)); % Sigmoid
end

function y = logsigd(x)
    y = x .* (1 - x); % Sigmoid derivative
end

function y = tansig(x)
    y = tanh(x); % Hyperbolic tangent
end

function y = tansigd(x)
    y = 1 - x.^2; % Tanh derivative
end
