function [Out_A] = annihilating(Out, events, ratio, min_filter_length)
    if nargin < 4 || isempty(min_filter_length)
        min_filter_length = 2;
    end

    stops = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99];
    len_Out = length(Out);
    fprintf('Processing with annihilating filter, ratio = %.4f\n', ratio);

    for l = 1:len_Out
        A = Out(l).A;
        y = Out(l).y;
        reconX = Out(l).x_reconstr;
        [rmse, stop_an, L_stop, stop_rmse] = stopLength(A, y, reconX, events);
        Out_A(l).rmse = rmse; Out_A(l).stop_an = stop_an;
        Out_A(l).L_stop = L_stop; Out_A(l).stop_rmse = stop_rmse;
        Out_A(l).A = A; Out_A(l).y = y;

        if ratio > 0 && ratio <= 1
            [~, idx] = min(abs(stops - ratio));
            tempL = L_stop(idx);
        elseif ratio > 1 && mod(ratio,1)==0 && ratio <= length(L_stop)
            tempL = L_stop(round(ratio));
        else
            candidate_L = round(ratio);
            max_L = min(length(reconX) - 5, 90);
            tempL = min(max(candidate_L, min_filter_length), max_L);
        end

        if isnan(tempL) || tempL >= length(reconX)-1
            tempL = max(min(floor(length(reconX)/3), 50), min_filter_length);
        end

        N = length(reconX);
        Xmat = zeros(N-tempL+1, tempL);
        for s = 1:N-tempL+1
            Xmat(s,:) = reconX(s:s+tempL-1);
        end

        Ymat = Xmat.' * Xmat;
        if cond(Ymat) > 1e10 || isinf(cond(Ymat))
            Ymat = Ymat + eye(size(Ymat)) * (norm(Ymat, 'fro') * 1e-4);
        end

        [U, ~, ~] = svd(Ymat);
        h = U(:,end);
        c1 = [h(1); zeros(N-tempL,1)];
        r1 = zeros(1,N); r1(1:tempL) = h.';
        H = toeplitz(c1,r1);

        combined_model = [A; H];
        combined_target = [y; zeros(N-tempL+1,1)];
        if cond(combined_model) > 1e10 || isinf(cond(combined_model))
            combined_model = combined_model + eye(size(combined_model)) * 1e-4;
        end

        xhat = (pinv(combined_model, 1e-6) * combined_target).';
        if any(isnan(xhat)) || any(isinf(xhat))
            xhat = zeros(1,N);
        end

        Out_A(l).x_reconstr = xhat';
        Out_A(l).Matrix = H;
        Out_A(l).muvars = Out(l).muvars;
        Out_A(l).error = sqrt(mean((xhat' - events).^2, 1));
        Out_A(l).h = h;
        % Store iteration number for future reference
        if isfield(Out(l), 'iteration')
            Out_A(l).iteration = Out(l).iteration + 1;
        else
            Out_A(l).iteration = 1;
        end

        Out_A(l).L = tempL;
    end
end
