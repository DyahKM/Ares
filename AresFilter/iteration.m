function [inc, Out_First, Out_Last] = iteration(Out, events, time, Gama, ratio)
    % Initialize error tracking arrays
    error = zeros(length(Out), 1);
    for i = 1:length(Out)
        if isfield(Out(i), 'error')
            error(i) = Out(i).error;
        else
            error(i) = NaN;
            fprintf('Warning: Missing error field in Out(%d)\n', i);
        end
    end

    Serror = zeros(length(Out), time);
    ActError = zeros(length(Out), time+1);
    ActError(:, 1) = error;
    error_TP = error;
    dummyOut = Out;

    % Store the Cost function result
    CF = zeros(time, length(Out));

    % Initialize previous reconstructions
    prev_recon = cell(1, length(Out));
    for i = 1:length(Out)
        if isfield(Out(i), 'x_reconstr')
            prev_recon{i} = Out(i).x_reconstr;
        else
            prev_recon{i} = zeros(size(events));
        end
        dummyOut(i).iteration = 0;
    end

    no_improvement_count = 0;

    for j = 1:time
        fprintf('\n===== Iteration %d =====\n', j);
        dummyTP = error_TP;

        % Adaptive gamma adjustment
        adaptive_gama = Gama;
        if no_improvement_count > 0
            adaptive_gama = Gama * (1 + 0.1 * ((-1)^j) * (j / time));
            fprintf('Using adaptive gamma: %.4f (original: %.4f)\n', adaptive_gama, Gama);
        end

        % Adjust ratio slightly to promote exploration
        current_ratio = ratio;
        if j > 1
            ratio_perturb = 0.05 * sin(j * pi/2);
            current_ratio = ratio * (1 + ratio_perturb);
            fprintf('Adjusting ratio to %.4f (base: %.2f)\n', current_ratio, ratio);
        end

        % Call annihilating filter
        tic
        [Out_A] = annihilating(dummyOut, events, current_ratio);
        toc

        % Update errors
        for i = 1:length(Out)
            error_TP(i) = Out_A(i).error;
        end
        ActError(:, j+1) = error_TP;

        % Track reconstruction change
        recon_change = 0;
        for i = 1:length(Out)
            if isfield(Out_A(i), 'x_reconstr') && ~any(isnan(Out_A(i).x_reconstr))
                recon_change = recon_change + norm(Out_A(i).x_reconstr - prev_recon{i}) / (norm(prev_recon{i}) + eps);
                prev_recon{i} = Out_A(i).x_reconstr;
            end
        end
        avg_recon_change = recon_change / length(Out);
        fprintf('Average relative change in reconstruction: %.6f\n', avg_recon_change);

        % Stagnation check
        if avg_recon_change < 1e-4
            no_improvement_count = no_improvement_count + 1;
            fprintf('WARNING: Minimal improvement detected (%d times)\n', no_improvement_count);

            if no_improvement_count >= 2
                fprintf('Applying perturbation to escape local minimum...\n');
                perturbation_scale = 0.01 * no_improvement_count;

                for i = 1:length(Out_A)
                    noise = perturbation_scale * randn(size(Out_A(i).x_reconstr));
                    Out_A(i).x_reconstr = Out_A(i).x_reconstr + noise;

                    if length(Out_A(i).x_reconstr) == length(events)
                        Out_A(i).error = sqrt(mean((Out_A(i).x_reconstr - events).^2));
                    end
                end

                fprintf('Applied %.2f%% perturbation\n', perturbation_scale*100);
            end
        else
            no_improvement_count = 0;
        end

        % Compute cost function (data fit + filter term)
        L = zeros(length(Out), 1);
        for l = 1:length(Out)
            xhat = Out_A(l).x_reconstr;

            if any(isnan(xhat)) || any(isinf(xhat))
                L(l) = Inf;
                continue;
            end

            data_term = norm(Out_A(l).y - Out_A(l).A * xhat, 2)^2;
            filter_term = norm(Out_A(l).Matrix * xhat, 2)^2;

            filter_weight = sqrt(adaptive_gama) * (1 + 0.1 * sin(j * pi/2));
            L(l) = sqrt(1 - adaptive_gama) * data_term + filter_weight * filter_term;

            fprintf('Config %d: Data term: %.4f, Filter term: %.4f, Cost: %.4f\n', ...
                    l, data_term, filter_term, L(l));
        end
        CF(j, :) = L;

        % Compare error changes
        ratio_err = dummyTP ./ (error_TP + eps);
        Serror(:, j) = double(ratio_err < 1) - double(ratio_err > 1);

        % Show improvement stats
        improved = sum(Serror(:, j) < 0);
        worsened = sum(Serror(:, j) > 0);
        same = sum(Serror(:, j) == 0);
        fprintf('Improvement stats: improved=%d, worsened=%d, same=%d\n', improved, worsened, same);
        fprintf('Current average RMSE: %.6f\n', mean(error_TP));

        dummyOut = Out_A;
        if j == 1
            Out_First = Out_A;
        end
    end

    Out_Last = Out_A;
    inc.ActError = ActError;
    inc.Serror = Serror;
    inc.Lagrangian = CF;

    % Final summary
    fprintf('\n===== Final Results =====\n');
    fprintf('Initial average RMSE: %.6f\n', mean(ActError(:, 1)));
    fprintf('Final average RMSE:   %.6f\n', mean(ActError(:, end)));
    improvement = (mean(ActError(:, 1)) - mean(ActError(:, end))) / mean(ActError(:, 1)) * 100;
    fprintf('Improvement:          %.2f%%\n', improvement);

    % Optional RMSE plot
    try
        figure;
        plot(0:time, mean(ActError, 1), 'b-o', 'LineWidth', 2);
        title('Average RMSE Progression');
        xlabel('Iteration');
        ylabel('RMSE');
        grid on;
    catch
        fprintf('Note: Unable to create plot (likely in non-GUI environment)\n');
    end
end
