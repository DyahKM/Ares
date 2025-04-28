function [inc,Out_First,Out_Last] = iteration(Out, events, time, Gama, ratio)
    % LorT = Length if using annihiCFting filter
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
	
    for j = 1:time
        fprintf('iteration %d\n',j);
        dummyTP = error_TP;
	
	tic	
        [Out_A] = annihilating(dummyOut, events, ratio);
        toc
        
        for i=1:length(Out)
			error_TP(i) = Out_A(i).error;
        end
        
        ActError(:,j+1) = error_TP;
		
        L = zeros(length(Out),length(Gama));
		
        for l = 1:length(Out)
            L(l) = norm(sqrt(1-Gama)*(Out_A(l).y-Out_A(l).A*Out_A(l).x_reconstr),2)+ ...
                norm(sqrt(Gama)*(Out_A(l).Matrix*Out_A(l).x_reconstr));
        end

        CF(j,:) = L;
		
		% Compare previous error with the current error. If previous smaller, then 1, else if previous larger then -1, else 0.
        Serror(:,j) = double((dummyTP./error_TP)<1)-double((dummyTP./error_TP)>1);
		
        dummyOut = Out_A;
        if j==1
            Out_First = Out_A;
        end
		
		
    end
    
    Out_Last = Out_A;
    
    inc.ActError = ActError;
    
    inc.Serror = Serror;
   	inc.Lagrangian = CF;

end
