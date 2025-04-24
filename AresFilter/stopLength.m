function [rmse, stop_an, L, stop_rmse] = stopLength(A, y, x, events)
    % Allows for a more diverse range of filter lengths based on stop ratios
    Length = 2:2:90;  % Changed from 1:1:90 to provide better spacing
    stop_an = zeros(length(Length),1);
    rmse = zeros(length(Length),1);

    len = length(x);
    
    % Calculate metrics for each possible filter length
    for i = 1:length(Length)
        Ltemp = Length(i);
        
        % Skip if filter length is too large for the data
        if Ltemp >= len-5
            stop_an(i) = Inf;
            rmse(i) = Inf;
            continue;
        end
        
        Xmat=zeros(len-Ltemp+1,Ltemp);
        for s=1:len-Ltemp+1
            Xmat(s,:)=x(s:s+Ltemp-1);
        end

        Ymat=Xmat.'*Xmat; % small LxL matrix
        
        % Check matrix condition before SVD
        if cond(Ymat) > 1e12
            fprintf('Warning: Matrix ill-conditioned for L=%d, cond=%e\n', Ltemp, cond(Ymat));
            stop_an(i) = Inf;
            rmse(i) = Inf;
            continue;
        end
        
        [U,Sigma,V]=svd(Ymat); % only need eigenvec corr to smallest eigenvalue
        h=U(:,end);
        c1=[h(1); zeros(len-Ltemp,1)];
        r1=zeros(1,len);
        r1(1:Ltemp)=h.';
        Han=toeplitz(c1,r1);

        Aan = [A;Han];
        yan = [y; zeros(len-Ltemp+1,1)];

        xhat_an = (pinv(Aan)*yan).';
        
        rmse(i) = sqrt(mean((events'-xhat_an).^2));
        
        % Modified cost function with better scaling of the regularization term
        stop_an(i) = norm((y-A*xhat_an'),2)^2 + ... % Data fidelity term
                    0.1 * norm((Han*xhat_an'),2)^2 + ... % Filter term (reduced weight)
                    0.01 * Ltemp; % Length regularization (much smaller weight)
    end
    
    % More diverse range of stops
    stops = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99];
    
    % Record the minimum value to compute the stopping criterion
    min_stop = min(stop_an);
    max_stop = max(stop_an(~isinf(stop_an)));
    diff = stop_an(1) - min_stop;
    
    % Initialize arrays to store L values and corresponding RMSE
    L = zeros(1, length(stops));
    stop_rmse = zeros(1, length(stops));
    
    % More dynamic method to select filter lengths based on stop_ratios
    for k = 1:length(stops)
        threshold = min_stop + diff * stops(k);
        valid_indices = find(stop_an <= threshold & ~isinf(stop_an));
        
        if isempty(valid_indices)
            fprintf('Warning: No valid filter length found for stop ratio %.2f\n', stops(k));
            % Use a default value based on the ratio
            L(k) = max(2, round(len * stops(k) / 10));
            stop_rmse(k) = NaN;
        else
            % Select the smallest L that satisfies the threshold
            [~, min_idx] = min(Length(valid_indices));
            L(k) = Length(valid_indices(min_idx));
            stop_rmse(k) = rmse(valid_indices(min_idx));
        end
        
        fprintf('Stop ratio %.2f -> Selected filter length L = %d (RMSE: %.4f)\n', stops(k), L(k), stop_rmse(k));
    end
    
    % Display a summary of unique L values selected
    unique_L = unique(L);
    fprintf('\nUnique filter lengths selected: ');
    fprintf('%d ', unique_L);
    fprintf('\n');
end
