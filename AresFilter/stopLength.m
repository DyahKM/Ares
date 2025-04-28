function [rmse, stop_an, L, stop_rmse] = stopLength(A, y, x, events)
    Length = [2:2:20, 22:3:40, 45:5:70, 75:10:90];
    stop_an = zeros(length(Length), 1);
    rmse = zeros(length(Length), 1);
    len = length(x);
    valid_lengths = 0;

    for i = 1:length(Length)
        Ltemp = Length(i);
        if Ltemp >= len-5
            stop_an(i) = Inf;
            rmse(i) = Inf;
            continue;
        end

        Xmat = zeros(len-Ltemp+1, Ltemp);
        for s = 1:len-Ltemp+1
            Xmat(s,:) = x(s:s+Ltemp-1);
        end

        Ymat = Xmat.' * Xmat;
        if cond(Ymat) > 1e10 || isinf(cond(Ymat))
            fprintf('Warning: Matrix ill-conditioned for L=%d, cond=%e — skipping\n', Ltemp, cond(Ymat));
            stop_an(i) = Inf;
            rmse(i) = Inf;
            continue;
        end

        [U, ~, ~] = svd(Ymat);
        h = U(:,end);
        c1 = [h(1); zeros(len-Ltemp,1)];
        r1 = zeros(1,len);
        r1(1:Ltemp) = h.';
        Han = toeplitz(c1,r1);

        Aan = [A; Han];
        yan = [y; zeros(len-Ltemp+1,1)];
        xhat_an = (pinv(Aan, 1e-6) * yan).';
        if any(isnan(xhat_an)) || any(isinf(xhat_an))
            stop_an(i) = Inf;
            rmse(i) = Inf;
            continue;
        end

        rmse(i) = sqrt(mean((events' - xhat_an).^2));
        data_fit = norm((y - A*xhat_an'), 2)^2;
        filter_smoothness = norm((Han*xhat_an'), 2)^2;
        length_penalty = 5 / Ltemp;  % penalize small L more
        length_factor = 0.1 + 0.9 * (Ltemp / max(Length));
        stop_an(i) = data_fit + (filter_smoothness * length_factor) + length_penalty;
        valid_lengths = valid_lengths + 1;
    end

    if all(rmse == 0)
        warning('All RMSE values are zero — likely overfit or invalid input.');
        rmse = rmse + rand(size(rmse)) * 1e-3;
    end

    stops = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99];
    if valid_lengths < 3
        L = min(10, floor(len/4)) * ones(1, length(stops));
        stop_rmse = NaN * ones(1, length(stops));
        return;
    end

    min_stop = min(stop_an(~isinf(stop_an)));
    max_stop = max(stop_an(~isinf(stop_an)));
    range_stop = max_stop - min_stop;
    L = zeros(1, length(stops));
    stop_rmse = zeros(1, length(stops));

    for k = 1:length(stops)
        threshold = min_stop + range_stop * stops(k);
        valid_indices = find(stop_an <= threshold & ~isinf(stop_an));
        if isempty(valid_indices)
            L(k) = max(2, min(round(len * stops(k) / 5), len-6));
            stop_rmse(k) = NaN;
        else
            if stops(k) < 0.3
                [~, min_idx] = min(Length(valid_indices));
                L(k) = Length(valid_indices(min_idx));
            elseif stops(k) > 0.7
                [~, max_idx] = max(Length(valid_indices));
                L(k) = Length(valid_indices(max_idx));
            else
                weights = 0.7 * (1 - rmse(valid_indices) / max(rmse(valid_indices))) + 0.3 * (Length(valid_indices) / max(Length));
                [~, best_idx_all] = max(weights);
                best_idx = best_idx_all(1);
                L(k) = Length(valid_indices(best_idx));
            end
            stop_rmse(k) = rmse(find(Length == L(k), 1));
        end
        fprintf('Stop ratio %.2f -> Selected filter length L = %d (RMSE: %.4f)\n', stops(k), L(k), stop_rmse(k));
    end
end
