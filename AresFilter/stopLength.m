function [inc,Out_First,Out_Last] = iteration(Out, events, time, Gama, ratio)
    % LorT = Length if using annihilating filter
    % LorT = Threshold if using fourier filter
    error = zeros(length(Out),1);
    for i=1:length(Out)
        error(i) = Out(i).error;
    end
    
    % Compare error sequence
    Serror = zeros(length(Out),time);
    
    % Actual error sequence
    ActError = zeros(length(Out),time+1);
    ActError(:,1) = error;
    
    error_TP = error;
    dummyOut = Out;
    
    % Store the Cost function result.
    CF = zeros(time, length(Out));
    
    % Store previous reconstruction to check for convergence
    prev_recon = cell(1, length(Out));
    for i = 1:length(Out)
        if isfield(Out(i), 'x_reconstr')
            prev_recon{i} = Out(i).x_reconstr;
        else
            prev_recon{i} = zeros(size(events));
        end
    end
    
    % Track if improvement stopped
    no_improvement_count = 0;
    
    for j = 1:time
        fprintf('\n===== Iteration %d =====\n', j);
        dummyTP = error_TP;
        
        % Adaptive gamma parameter to avoid local minima
        adaptive_gama = Gama;
        if no_improvement_count > 0
            % Perturb Gama if we've stalled
            adaptive_gama = Gama * (1 + 0.1 * ((-1)^j) * (j/time));
            fprintf('Using adaptive gamma: %.4f (original: %.4f)\n', adaptive_gama, Gama);
        end
        
        % Log the stop ratio being used
        fprintf('Using stop ratio: %.2f\n', ratio);
        
        tic    
        % Call the annihilating filter function
        [Out_A] = annihilating(dummyOut, events, ratio);
        toc
        
        % Calculate new errors
        for i=1:length(Out)
            error_TP(i) = Out_A(i).error;
        end
        
        ActError(:,j+1) = error_TP;
        
        % Calculate change in reconstruction
        recon_change = 0;
        for i = 1:length(Out)
            if isfield(Out_A(i), 'x_reconstr')
                recon_change = recon_change + norm(Out_A(i).x_reconstr - prev_recon{i}) / norm(prev_recon{i});
                prev_recon{i} = Out_A(i).x_reconstr;
            end
        end
        avg_recon_change = recon_change / length(Out);
        fprintf('Average relative change in reconstruction: %.6f\n', avg_recon_change);
        
        % Check for stagnation
        if avg_recon_change < 1e-4
            no_improvement_count = no_improvement_count + 1;
            fprintf('WARNING: Minimal improvement detected (%d times)\n', no_improvement_count);
            
            % Apply perturbation after consecutive stagnations
            if no_improvement_count >= 2
                fprintf('Applying perturbation to escape local minimum...\n');
                perturbation_scale = 0.01 * no_improvement_count;
                
                for i=1:length(Out_A)
                    % Small perturbation to the reconstruction
                    noise = perturbation_scale * randn(size(Out_A(i).x_reconstr));
                    Out_A(i).x_reconstr = Out_A(i).x_reconstr + noise;
                    
                    % Recalculate error with perturbed reconstruction
                    if length(Out_A(i).x_reconstr) == length(events)
                        Out_A(i).error = sqrt(mean((Out_A(i).x_reconstr - events).^2));
                    end
                end
                
                fprintf('Applied %.2f%% perturbation\n', perturbation_scale*100);
            end
        else
            no_improvement_count = 0;
        end
        
        % Calculate cost function with possibly adaptive gamma
        L = zeros(length(Out),1);
        for l = 1:length(Out)
            data_term = norm(Out_A(l).y - Out_A(l).A * Out_A(l).x_reconstr, 2)^2;
            filter_term = norm(Out_A(l).Matrix * Out_A(l).x_reconstr, 2)^2;
            
            L(l) = sqrt(1-adaptive_gama) * data_term + sqrt(adaptive_gama) * filter_term;
            
            fprintf('Config %d: Data term: %.4f, Filter term: %.4f, Cost: %.4f\n', ...
                   l, data_term, filter_term, L(l));
        end

        CF(j,:) = L;
        
        % Compare previous error with the current error
        % If previous smaller, then 1, else if previous larger then -1, else 0
        Serror(:,j) = double((dummyTP./error_TP)<1) - double((dummyTP./error_TP)>1);
        
        % Display improvement statistics
        improved = sum(Serror(:,j) < 0);
        worsened = sum(Serror(:,j) > 0);
        same = sum(Serror(:,j) == 0);
        fprintf('Improvement stats: improved=%d, worsened=%d, same=%d\n', improved, worsened, same);
        
        % Log current RMSE
        avg_rmse = mean(error_TP);
        fprintf('Current average RMSE: %.6f\n', avg_rmse);
        
        dummyOut = Out_A;
        if j==1
            Out_First = Out_A;
        end
    end
    
    Out_Last = Out_A;
    
    inc.ActError = ActError;
    inc.Serror = Serror;
    inc.Lagrangian = CF;
    
    % Final statistics
    fprintf('\n===== Final Results =====\n');
    fprintf('Initial average RMSE: %.6f\n', mean(ActError(:,1)));
    fprintf('Final average RMSE:   %.6f\n', mean(ActError(:,end)));
    improvement = (mean(ActError(:,1)) - mean(ActError(:,end))) / mean(ActError(:,1)) * 100;
    fprintf('Improvement:          %.2f%%\n', improvement);
    
    % Display error progression
    fprintf('\nRMSE progression:\n');
    for j = 1:time+1
        fprintf('Iteration %d: %.6f\n', j-1, mean(ActError(:,j)));
    end
end
