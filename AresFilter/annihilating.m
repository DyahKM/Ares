function [Out_A] = annihilating(Out, events, ratio, min_filter_length)
    % ANNIHILATING Applies annihilating filter to reconstructed signal
    % 
    % Inputs:
    %   Out - Structure with reconstruction data
    %   events - Original event data for error calculation
    %   ratio - Stop ratio parameter (can be index or value between 0-1)
    %   min_filter_length - Optional minimum filter length (default: 2)
    %
    % Outputs:
    %   Out_A - Structure with results after applying annihilating filter
    
    len_Out = length(Out);
    
    % Set default minimum filter length if not provided
    if nargin < 4 || isempty(min_filter_length)
        min_filter_length = 2;
    end
    
    fprintf('Processing with annihilating filter, ratio = %f\n', ratio);
    
    for l = 1:len_Out
        reconX = Out(l).x_reconstr;
        A = Out(l).A;
        y = Out(l).y;
        
        % Calculate potential filter lengths and associated metrics
        [rmse, stop_an, L_stop, stop_rmse] = stopLength(A, y, reconX, events);
        Out_A(l).rmse = rmse;
        Out_A(l).stop_an = stop_an;
        Out_A(l).L_stop = L_stop;
        Out_A(l).stop_rmse = stop_rmse;
        
        Out_A(l).A = A;
        Out_A(l).y = y;
        
        % Enhanced logic for selecting filter length based on ratio
        if isnumeric(ratio)
            % Define stops array directly here to match stopLength.m
            stops = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99];
            
            if ratio > 0 && ratio <= length(L_stop)
                % If ratio is an integer index, use it directly
                tempL = L_stop(round(ratio));
                fprintf('Using L_stop index %d = %d\n', round(ratio), tempL);
            elseif ratio > 0 && ratio <= 1
                % If ratio is a value between 0-1, find closest match in stops
                [~, idx] = min(abs(stops - ratio));
                tempL = L_stop(idx);
                fprintf('Using stop ratio %.2f, found match at index %d, L = %d\n', ratio, idx, tempL);
            else
                % For ratios > 1 but not integers, interpret as direct filter length
                % This adds more diversity to the filter length selection
                candidate_L = round(ratio);
                
                % Ensure it's within reasonable bounds
                max_L = min(length(reconX) - 5, 90);
                tempL = min(max(candidate_L, min_filter_length), max_L);
                
                fprintf('Using direct filter length L = %d (from ratio %.2f)\n', tempL, ratio);
            end
        else
            % Default to middle value if ratio is invalid
            tempL = L_stop(ceil(length(L_stop)/2));
            fprintf('Warning: Invalid ratio type, using default filter length L = %d\n', tempL);
        end
        
        % Ensure filter length is reasonable
        N = length(reconX);
        if tempL >= N-1
            fprintf('Warning: Filter length %d too large for signal length %d\n', tempL, N);
            tempL = max(min(floor(N/3), 50), min_filter_length); 
            fprintf('Adjusted to L = %d\n', tempL);
        end
        
        % Apply a small perturbation to filter length based on iteration 
        % This helps avoid homogeneity in results
        if isfield(Out(l), 'iteration') && ~isempty(Out(l).iteration)
            iteration = Out(l).iteration;
            perturb = (-1)^iteration * mod(iteration, 3);
            tempL = max(tempL + perturb, min_filter_length);
            fprintf('Applied perturbation for iteration %d, new L = %d\n', iteration, tempL);
        end
        
        % Store the selected filter length
        Out_A(l).L = tempL;
        
        % Apply the annihilating filter
        reconX = reconX.';
        Xmat = zeros(N-tempL+1, tempL);

        for s = 1:N-tempL+1
            Xmat(s,:) = reconX(s:s+tempL-1);
        end

        % Calculate filter coefficients using SVD
        Ymat = Xmat.' * Xmat; 
        
        % Check matrix condition before SVD
        if cond(Ymat) > 1e12
            fprintf('Warning: Matrix ill-conditioned for sample %d, L=%d, cond=%e\n', l, tempL, cond(Ymat));
            
            % Add small regularization to improve conditioning
            Ymat = Ymat + eye(size(Ymat)) * (norm(Ymat, 'fro') * 1e-10);
        end
        
        [U, Sigma, V] = svdecon(Ymat);
        h = U(:,end);
        
        % Construct annihilating filter matrix
        c1 = [h(1); zeros(N-tempL,1)];
        r1 = zeros(1,N);
        r1(1:tempL) = h.';
        H = toeplitz(c1,r1);

        % Apply the combined model (measurement + filter constraints)
        % Add small regularization to improve numerical stability
        combined_model = [A; H];
        combined_target = [y; zeros(N-tempL+1,1)];
        
        % Use pinv with small tolerance for better numerical stability
        xhat = (pinv(combined_model, 1e-8) * combined_target).';
        
        % Calculate error compared to ground truth
        error = sqrt(mean((xhat' - events).^2, 1));
        
        % Store results
        Out_A(l).muvars = Out(l).muvars;
        Out_A(l).Matrix = H;
        Out_A(l).x_reconstr = xhat';
        Out_A(l).error = error;
        Out_A(l).h = h;
        
        % Store iteration number for future reference if provided
        if isfield(Out(l), 'iteration')
            Out_A(l).iteration = Out(l).iteration + 1;
        else
            Out_A(l).iteration = 1;
        end
        
        fprintf('Sample %d: Using filter length L = %d, RMSE = %.4f\n', l, tempL, error);
    end
    
    % Report overall results
    errors = zeros(1, len_Out);
    for l = 1:len_Out
        errors(l) = Out_A(l).error;
    end
    fprintf('Overall mean RMSE: %.4f (min: %.4f, max: %.4f)\n', mean(errors), min(errors), max(errors));
    
end
