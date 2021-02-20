% Obtain the Q table of a bridge
clc
clear

% Basic Information of Q learning
episode = 2000;  % iteration numbers
eps = 0.5;   % epsilon for e-greedy search
alpha = 0.5;   % update steps
gamma = 0.9;   % discount factor

% Planning horizon
N = 50;    % Year  <<---------------------------------------<<<
delta = 2;   % Inspection interval

% WPHM parameters
beta = 1.1063;   % beta
ita = 0.127;   % ita
weight = -1.566;  % weight of CR

% Transition probaiblities of repairable condition rating i:<<-------------------------------------<<<
     % 9     8     7     6     5     4     3
P = [0.895,0.088,0.015,0.002,  0,    0,    0;
       0,  0.904,0.079,0.014,0.002,0.001,  0;
       0,    0,  0.923,0.064,0.011,0.002,  0; 
       0,    0,    0,  0.924,0.066,0.006,0.004;
       0,    0,    0,    0,  0.932,0.056,0.012;
       0,    0,    0,    0,    0,  0.956,0.044;
       0,    0,    0,    0,    0,    0,    1];
P_sum = cumsum(P,2);
pf = 0.0345;       % Failure probability if superstructure state is in (n,U)
Ncr = size(P,1);   % Number of repairable condition ratings

% Compound states
% Order: (0,9),(0,8),...,(0,3),(0,U),(1,9),(1,8),...,(1,3),(1,U),...,(100,9),(100,8),...,(100,3),(100,U)
state_num = (Ncr+1)*N;   % Number of states

% Actions: 1-No action; 2-Preventive repair; 3-Reconstruction
action_num = 3;   % Number of actions
% Preventive repair effect matrix - PREM (One-year transition probability)
      % 9     8     7     6     5     4     3 
PREM = [0,    0,    0,    0,    0,    0,    0;  % 9
        1,    0,    0,    0,    0,    0,    0;  % 8
      0.139,0.861,  0,    0,    0,    0,    0;  % 7
      0.064,0.209,0.727,  0,    0,    0,    0;  % 6
      0.046,0.088,0.135,0.731,  0,    0,    0;  % 5
      0.024,0.134,0.035,0.178,0.629,  0,    0;  % 4
      0.038,0.179,0.028,0.130,0.352,0.273,  0;  % 3 
        0,    0,  0.066,0.067,  0,   0.2,0.667]; % U
PREM_sum = cumsum(PREM,2);
    
R = NY_Rikr(beta,ita,weight,P,Ncr,delta,N);  % Survival function
N0 = N+delta;   % Actually, for decision-making at year N, we should consider the bridge performance in the next inspection interval.



% Basic information of the bridge
Condition_Rating = 8; % condition rating
Working_Age = 10; % Working age
s0 = Working_Age*(Ncr+1) + (10 - Condition_Rating); % Initial state of the bridge

% Calculate reward table
Magnitude = 42.4*25.2;   % Bridge magnitude (L*W) (m2)------------------------------------------------<<<
Traffic = 18479;    % Traffic volume (veh/d) ------------------------------------------------<<<
Detour = 8;   % Detour length (km) ------------------------------------------------<<<
pc = 0.94; pt = 0.06;   % Percent of cars and trucks ------------------------------------------<<<
Fuelc = 10; Fuelt = 40;   % Fuel consumption of cars and trucks (L/km) --------ccc
Pricef = 251.05; Pricef = Pricef * 0.2642 * 0.01;  % Fuel price ($/L) (present value) ---------ccc

Cre = 4921000; % Reconstruction cost -------------------------<<<
Cdt = Traffic*365*pt*Fuelt*Detour*Pricef;  % Traffic detour cost (assume half close, consider trucks only) -------------------------<<<
% Failure leads to 2-year traffic detour
Ch = pf*(Cre+2*Cdt);   % Annual holding cost
cI = 7000;  % Inspection cost ($) ------------------------------------------<<<
cM = (Cre+Cdt)*cI;   % Big M (Act as constraint)
% CR = 9   8    7    6    5    4    3    U
Cr = [cM,2513,2701,2731,2426,2335,2355,2340]*Magnitude;    % Cost of a preventive repair ($) ----------------------------------------<<<
% Order of S in cost matrix:(0,9)...(0,3)(0,U)...(1,x)...(N,9)...(N,3)(N,U)
% Row 1: ac = 0: do nothing
% Row 2: ac = 1: Preventive Repair
Reward = zeros(state_num,3);  % Initialization --------------------------<<<
for s = 1:state_num   % Loop state s
    n = floor((s-1)/(Ncr+1));   % year
    i = mod((s-1),Ncr+1)+1;   % Obtain the deck condition rating i or U (actual condition rating =10-i)
    if i<=Ncr   % S1
        % No action
        pki = zeros(1,Ncr*N0); pki(Ncr*n+i) = 1;   % pi vector
        Rik2 = 1- pki*(eye(length(R))-R^2)*ones(Ncr*N0,1);   % Conditional that the bridge has survived for k years, and its current condition rating is i, the survival probability of the bridge in the next r years
        Reward(s,1) = cI*Rik2 + (Ch*delta*(1-Rik2))*(1-Rik2);
        % Preventive repair
        Exp_cost = 0; 
        for cr = 1:Ncr
            %Rik1 = NY_Rikr(beta,ita,weight,P,Ncr,delta,N,cr,n+1,1);
            pki = zeros(1,Ncr*N0); pki(Ncr*(n+1)+cr) = 1;   % pi vector
            Rik1 = 1- pki*(eye(length(R))-R^1)*ones(Ncr*N0,1);
            Exp_cost = Exp_cost + PREM(i,cr)*(cI*Rik1 + (Ch*delta*(1-Rik1))*(1-Rik1));
        end
        Reward(s,2) = Cr(i) + Exp_cost;
        % Reconstruction repair
        Reward(s,3) = Cre;
    else   % S2
        % No action
        Reward(s,1) = cM;   % It must be reconstructed if it is unsatisfied
        % Preventive repair
        Exp_cost = 0;
        for cr = 1:Ncr
            %Rik1 = NY_Rikr(beta,ita,weight,P,Ncr,delta,N,cr,n+1,1);
            pki = zeros(1,Ncr*N0); pki(Ncr*(n+1)+cr) = 1;   % pi vector
            Rik1 = 1- pki*(eye(length(R))-R^1)*ones(Ncr*N0,1);
            Exp_cost = Exp_cost + PREM(i,cr)*(cI*Rik1 + (Ch*delta*(1-Rik1))*(1-Rik1));
        end
        Reward(s,2) = Cr(i) + Exp_cost;
        % Reconstruction repair
        Reward(s,3) = Cre;
    end
end

% Q learning
Q_table = zeros(state_num,action_num); % Initialize Q(s,a)
Q_table0 = zeros(state_num,action_num);
Error = [];

% Update Q table
for i = 1:episode
    current_state = s0; %randperm(8,1); % randperm(state_num,1);   % randomly choose a state as initial state
    itr = 0;    % number of iteration in each episode
     
    while current_state < (Ncr+1)*(N-1)
        t = floor((current_state-1)/(Ncr+1));   % Obtain year
        CR = mod((current_state-1),Ncr+1)+1;   % Obtain the deck condition rating i or U (Note: actual condition rating = 10-CR)
        ava_action = find(Reward(current_state,:)<cM);    % Select available action from reward table
        % epsilon-greedy search
        if rand()<eps
            chosen_action = ava_action(randperm(length(ava_action),1));   % randomly select an action from ALL actions
        else
            % randomly choose an action from current state
            chosen_action = find(Reward(current_state,:) == min(Reward(current_state,:)));    % Select the action resulting in the min cost
        end
        % take action, observe reward and next state
        r = Reward(current_state,chosen_action);    % Reward
        
        % transition to the next state after taking the action
        switch chosen_action
            % if taking action 1 - no action
            case 1
                a1 = rand(1);
                j = 1;    % start from CR = 9
                while a1>P_sum(CR,j)
                    j = j + 1;
                end
                %CR = j;
                next_state = (t+1)*(Ncr+1) + j;
            % if taking action 2 - preventive repair
            case 2
                a1 = rand(1);
                j = 1;    % start from CR = 9
                while a1>PREM_sum(CR,j)
                    j = j + 1;
                end
                %CR = j;
                next_state = (t+1)*(Ncr+1) + j;
            % if taking action 3 - reconstruction
            case 3
                %CR = 1; t = 0; 
                next_state = 1;    % to state (0,9)
        end

        % update Q-table
        maxQ = max(Q_table(next_state,:));
        Q_table(current_state,chosen_action) = Q_table(current_state,chosen_action) + alpha*(r + gamma*maxQ - Q_table(current_state,chosen_action));
        current_state = next_state;
        itr = itr + 1;
    end
    Error = [Error;mean(max(abs(Q_table0 - Q_table)))];
    Q_table0 = Q_table;
end

% Plot convergence
plot(1:length(Error),Error,'r-')
