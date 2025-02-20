function Compare_Load_Flow(sd)


    % Load the first 1000 columns of a CSV file
    %filename = 'C:\Users\robot\OpenDSS_Test\Voltage.csv'; % Replace with your CSV file name
    %data = readmatrix(filename); % Load the CSV file into a matrix
    
    
    %data = data(2:end, 2:1000);
    
    load('est_results.mat')
    load('G_DSS.mat')
    
    
    customers = double(reshape(all_loads', 1, []));
    %plot(array(:, find(customers~=0)));
    
    
    %figure()
    All_Loads = cell2mat(Node_Loads);
    pA = All_Loads(cell2mat(Node_Phases)==1);
    pB = All_Loads(cell2mat(Node_Phases)==2);
    pC = All_Loads(cell2mat(Node_Phases)==3);

    %{
    subplot(1, 3, 1)
    plot(array(:, find(ismember(customers, pA))))
    %ylim([210, 240])
    
    subplot(1, 3, 2)
    plot(array(:, find(ismember(customers, pB))))
    %ylim([210, 240])
    
    subplot(1, 3, 3)
    plot(array(:, find(ismember(customers, pC))))
    %ylim([210, 240])
    %}
    
    randomValues = sd * randn(size(array(:, 6:end)));
    array(:, 6:end) = array(:, 6:end) + randomValues;
    
    Phase_A = array(:, find(ismember(customers, pA)));
    Phase_B = array(:, find(ismember(customers, pB)));
    Phase_C = array(:, find(ismember(customers, pC)));
    
    
    
    pABC_Nodes = {[], [], []};
    
    %figure
    shortest_dist = [];
    hold on
    for phase=1:3
        pA_dist_avg = [];
        for i=1:length(Node_Phases)
            if length(Node_Phases{i}) > 0
                if any(ismember(Node_Phases{i}, [phase]))
                    node = Node_Loads{i}(ismember(Node_Phases{i}, [phase]));
                    d = distances(G,  height(G.Nodes), i);
                    pA_dist_avg = [pA_dist_avg; [d, max(array(:, customers==node) ./ array(:, 3+phase) )]];

                    pABC_Nodes{phase} = [pABC_Nodes{phase}, i];

                end
            end
        end
        
        [~, index] = min(pA_dist_avg(:, 1));
        %scatter(pA_dist_avg(:, 1), pA_dist_avg(:, 2))
        shortest_dist = [shortest_dist, index];
    end
    
    
    scriptvar = {};
    
    scriptvar.V1 = Phase_A;
    scriptvar.IR1 = 1000*P_profile(:, pA) ./ Phase_A;
    scriptvar.IX1 = 1000*Q_profiles(:, pA) ./ Phase_A;
    
    scriptvar.V2 = Phase_B;
    scriptvar.IR2 = 1000*P_profile(:, pB) ./ Phase_B;
    scriptvar.IX2 = 1000*Q_profiles(:, pB) ./ Phase_B;
    
    scriptvar.V3 = Phase_C;
    scriptvar.IR3 = 1000*P_profile(:, pC) ./ Phase_C;
    scriptvar.IX3 = 1000*Q_profiles(:, pC) ./ Phase_C;
    
    
    
    
    scriptvar.V1 = scriptvar.V1';
    scriptvar.V2 = scriptvar.V2';
    scriptvar.V3 = scriptvar.V3';
    scriptvar.AMI_used_1 = pA';
    
    scriptvar.IR1 = scriptvar.IR1';
    scriptvar.IR2 = scriptvar.IR2';
    scriptvar.IR3 = scriptvar.IR3';
    scriptvar.AMI_used_2 = pB';
    
    scriptvar.IX1 = scriptvar.IX1';
    scriptvar.IX2 = scriptvar.IX2';
    scriptvar.IX3 = scriptvar.IX3';
    scriptvar.AMI_used_3 = pC';
    
    scriptvar.V = [scriptvar.V1; scriptvar.V2; scriptvar.V3];
    scriptvar.IR = [scriptvar.IR1; scriptvar.IR2; scriptvar.IR3];
    scriptvar.IX = [scriptvar.IX1; scriptvar.IX2; scriptvar.IX3];
    
    scriptvar.invalid = zeros(1, length(scriptvar.V1))';
    scriptvar.invalid1 = zeros(1, length(scriptvar.V1))';
    scriptvar.invalid2 = zeros(1, length(scriptvar.V1))';
    scriptvar.invalid3 = zeros(1, length(scriptvar.V1))';

    scriptvar.Tr_PQ = Tr_PQ;
    
    scriptvar.AMI_used_ = [];

    scriptvar.shortest_dist = shortest_dist;
    scriptvar.ABC_Nodes = pABC_Nodes;

    save("Synthetic_data", "scriptvar")

end