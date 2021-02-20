% Calculate D matrix, Lambda Matrix. Return R(r|k,i)
% Input: beta,ita,y,TPM,Total No.of CR,Inspection Interval,Planning Horizon,Current CR, Survival years,Aim years
function R = NY_Rikr(beta,ita,y,P,Ncr,delta,N) %,CR,k,r)
% Assembly D matrix
Z8x = zeros(Ncr);    % Zero matrix
one_vector = ones(Ncr,1);   % one vector (Ncrx1)
N0 = N+delta;   % Actually, for decision-making at year N, we should consider the bridge performance in the next inspection interval.
D = [];
for K = 0:(N0-1)   % Row loop
    % Calculate Lambda(ij)(k) - Lij(k)
    L = zeros(Ncr);   % Initialization Lambda matrix in year k
    for i = 1:Ncr
        for j = 1:Ncr
            if i<=j
                x1 = 10-i;   % Condition rating in year k (The highest is 9)
                % integral of h(t,x) form k to k+1
            	dk = 0.001;     % mesh time
                H = 0;
                for tk = K:dk:(K+1)
                    h = beta/ita*(tk/ita)^(beta-1)*exp(y*x1)*dk; %----------------------!!!
                    H = H + h;
                end
                % Calculate Lambda(ij)
                L(i,j) = P(i,j)*exp(-H);
            end
        end
    end
    
    % Accumulate row onto D matrix
    if K < N0-1
        Dr = [];   % Initialization row (2nd~Nth sections)
        for i = 1:(N0-1)
            if i == K+1
                Dr = [Dr,L];
            else
                Dr = [Dr,zeros(Ncr)];
            end
        end
        Dr = [zeros(Ncr),Dr,(eye(Ncr)-L)*one_vector];
        D = [D;Dr];
    else
        D = [D;[zeros(Ncr,Ncr*(N0-1)),L,(eye(Ncr)-L)*one_vector]];
    end
end
D = [D;[zeros(1,N0*Ncr),1]];   % The last row of D Matrix
R = D(1:length(D)-1,1:length(D)-1);
end