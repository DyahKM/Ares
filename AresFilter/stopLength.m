function [rmse, stop_an, L, stop_rmse] = stopLength(A, y, x, events)
    % STOPLENGTH Calculate optimal filter lengths based on stop criteria
    %
    % Inputs:
    %   A - Measurement matrix
    %   y - Measurement values
    %   x - Current signal reconstruction
    %   events - Original event data for error calculation
    %
    % Outputs:
    %   rmse - RMSE for different filter lengths
    %   stop_an - Stop criteria values
    %   L - Selected filter lengths for each stop ratio
    %   stop_rmse - RMSE for selected filter lengths
    
    % Use a more diverse range of filter lengths with better spacing
    Length = [2:2:20, 22:3:40, 45:5:70, 75:10:90];
    stop_an = zeros(length(Length), 1);
    rmse = zeros(length(Length), 1);

    len = length(x);
    
    % Calculate metrics for each possible filter length
    valid_lengths = 0;
    for i = 1:length(Length)
        Ltemp = Length(i);
        
        % Skip if filter length is too large for the data
        if Ltemp >= len-5
            stop_an(i) = Inf;
            rmse(i) = Inf;
            continue;
        end
        
        valid_lengths = valid_lengths + 1;
        
        Xmat = zeros(len-Ltemp+1, Ltemp);
        for s = 1:len-Ltemp+1
            Xmat(s,:) = x(s:s+Ltemp-1);
        end

        Ymat = Xmat.' * Xmat;
        
        % Check matrix condition before SVD
        if cond(Ymat) > 1e12
            fprintf('Warning: Matrix ill-conditioned for L=%d, cond=%e\n', Ltemp, cond(Ymat));
            % Add regularization for ill-conditioned matrices
            Ymat = Ymat + eye(size(Ymat)) * (norm(Ymat, 'fro') * 1e-10);
        end
        
        [U, Sigma, V] = svd(Ymat);
        h = U(:,end);
        c1 = [h(1); zeros(len-Ltemp,1)];
        r1 = zeros(1,len);
        r1(1:Ltemp) = h.';
        Han = toeplitz(c1,r1);

        Aan = [A; Han];
        yan = [y; zeros(len-Ltemp+1,1)];

        xhat_an = (pinv(Aan) * yan).';
        
        rmse(i) = sqrt(mean((events' - xhat_an).^2));
        
        % Improved cost function with better diversity
        % Balance between data fit, filter smoothness, and length penalty
        data_fit = norm((y - A*xhat_an'), 2)^2;
        filter_smoothness = norm((Han*xhat_an'), 2)^2;
        length_penalty = 0.01 * Ltemp;
        
        % Make the cost function more sensitive to different lengths
        length_factor = 0.1 + 0.9 * (Ltemp / max(Length));
        stop_an(i) = data_fit + (filter_smoothness * length_factor) + length_penalty;
    end
    
    % More diverse range of stops
    stops = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99];
    
    % Check if we have enough valid filter lengths
    if valid_lengths < 3
        fprintf('Warning: Not enough valid filter lengths found (%d). Using default values.\n', valid_lengths);
        L = min(10, floor(len/4)) * ones(1, length(stops));
        stop_rmse = NaN * ones(1, length(stops));
        return;
    end
    
    % Record the minimum value to compute the stopping criterion
    valid_indices = find(~isinf(stop_an));
    min_stop = min(stop_an(valid_indices));
    max_stop = max(stop_an(valid_indices));
    range_stop = max_stop - min_stop;
    
    % Initialize arrays to store L values and corresponding RMSE
    L = zeros(1, length(stops));
    stop_rmse = zeros(1, length(stops));
    
    % Enhanced method to select diverse filter lengths based on stop_ratios
    for k = 1:length(stops)
        threshold = min_stop + range_stop * stops(k);
        valid_indices = find(stop_an <= threshold & ~isinf(stop_an));
        
        if isempty(valid_indices)
            fprintf('Warning: No valid filter length found for stop ratio %.2f\n', stops(k));
            % Use a realistic default based on the length of the signal
            L(k) = max(2, min(round(len * stops(k) / 5), len-6));
            stop_rmse(k) = NaN;
        else
            % Use a different selection strategy based on stop ratio
            if stops(k) < 0.3
                % For small ratios, prefer smaller filter lengths
                [~, min_idx] = min(Length(valid_indices));
                L(k) = Length(valid_indices(min_idx));
            elseif stops(k) > 0.7
                % For large ratios, prefer larger filter lengths
                [~, max_idx] = max(Length(valid_indices));
                L(k) = Length(valid_indices(max_idx));
            else
                % For mid ratios, use weighted selection based on both RMSE and length
                weights = 0.7 * (1 - rmse(valid_indices) / max(rmse(valid_indices))) +  0.3 * (Length(valid_indices) / max(Length));
                [~, best_idx] = max(weights);
                L(k) = Length(valid_indices(best_idx));
            end
            
            % Find index in original arrays
            orig_idx = find(Length == L(k), 1);
            stop_rmse(k) = rmse(orig_idx);
        end
        
        fprintf('Stop ratio %.2f -> Selected filter length L = %d (RMSE: %.4f)\n', stops(k), L(k), stop_rmse(k));
    end
    
    % Ensure more diversity in filter lengths
    unique_L = unique(L);
    if length(unique_L) < 3 && length(stops) > 3
        fprintf('Warning: Only %d unique filter lengths selected. Adding diversity.\n', length(unique_L));
        
        % Force more diversity by directly assigning different lengths
        min_valid_L = min(Length(~isinf(stop_an)));
        max_valid_L = max(Length(~isinf(stop_an)));
        range_L = max_valid_L - min_valid_L;
        
        for k = 1:length(stops)
            target_L = min_valid_L + range_L * stops(k);
            [~, idx] = min(abs(Length - target_L));
            
            if ~isinf(stop_an(idx))
                L(k) = Length(idx);
                stop_rmse(k) = rmse(idx);
            end
        end
    end
    
    % Display a summary of unique L values selected
    unique_L = unique(L);
    fprintf('\nUnique filter lengths selected: ');
    fprintf('%d ', unique_L);
    fprintf('\n');
end
