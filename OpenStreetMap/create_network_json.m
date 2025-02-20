function network_json = create_network_json(G)
    % Creates a JSON representation of the network from the provided tables
    % Args:
    %     df1: Table with node types (last column indicates customer=0, pole=1)
    %     df2: Table with edge list (FROM, TO, R, X columns)
    
    % Create the base structure using nested structs
    network_json = struct();
    network_json.LVNetwork = struct();
    
    % Add Circuit information
    network_json.LVNetwork.Circuit = struct(...
        'name', 'Realistic_LV_Circuit', ...
        'basekv', 0.4, ...
        'pu', 1.0, ...
        'phases', 3, ...
        'bus1', 'sourcebus' ...
    );
    
    % Add Transformer information
    network_json.LVNetwork.Transformer = struct(...
        'name', 'LV_Transformer', ...
        'kV', 0.4, ...
        'kVA', 150, ...
        'phases', 3, ...
        'XHL', 0.02, ...
        'windings', 2 ...
    );
    
    % Initialize empty arrays for Lines and Loads
    network_json.LVNetwork.Lines = struct([]);
    network_json.LVNetwork.Loads = struct([]);
    
    % Add Simulation Parameters
    network_json.SimulationParameters = struct(...
        'mode', 'daily', ...
        'stepsize', '30m', ...
        'duration_hours', 24 ...
    );
    
    % Add lines from df2
    lines_array = struct([]);

    phases = [G.Edges.pA, G.Edges.pB, G.Edges.pC];
    phase_13 = floor(sum(phases,2)/3);
    
    for idx = 1:height(G.Edges)

        n1 = G.Edges.EndNodes(idx, 1);
        n2 = G.Edges.EndNodes(idx, 2);

        % Multiphase
        phase_connects = find(phases(idx, :));

        if length(phase_connects)==3
            phase_connects = [0];
        end
        for p_c=1:length(phase_connects)

            line = struct(...
                'name', sprintf('Line%d', length(lines_array)+1), ...
                'bus1', sprintf('B_%d', n1), ...
                'bus2', sprintf('B_%d', n2), ...
                'length_km', G.Edges.Weight(idx), ...  % Default value as not provided
                'phases', phase_13(idx), ...
                'R1', 0.1, ...
                'X1', 0.2, ...
                'p_connect', phase_connects(p_c) ...
            );
            
            if idx == 1
                lines_array = line;
            else
                lines_array(length(lines_array)+1) = line;
            end

        end
    end
    network_json.LVNetwork.Lines = lines_array;
    
    % Add loads from df1 (where last column is 0)
    load_count = 1;
    loads_array = struct([]);

    Node_Loads = {};
    Node_Phases = {};
    
    for idx = 1:height(G.Nodes)
        % Get the value from the last column

        this_node = [];
        this_phases = [];

        if degree(G, idx)==1 && idx~=1  % Check if it's a customer (degree 1 and not start)
            phase_count = phase_13(findedge(G, idx, neighbors(G, idx)));
            spec_phase = find(phases(findedge(G, idx, neighbors(G, idx)), :));
            
            for p_c = 1:length(spec_phase)
                load = struct(...
                    'customer', sprintf('Load%d', load_count), ...
                    'bus', sprintf('B_%d', idx), ...
                    'phases', 1, ...
                    'kV', 0.4, ...
                    'kVAR', 0.15, ...  % Default value as not provided
                    'pf', 0.95, ...
                    'Vminpu', 0.94, ...
                    'Vmaxpu', 1.10, ...
                    'phase_connect', spec_phase(p_c) ...
                );
                
                if load_count == 1
                    loads_array = load;
                else
                    loads_array(load_count) = load;
                end

                this_node = [this_node, load_count];
                this_phases = [this_phases, spec_phase(p_c)];
                
                load_count = load_count + 1;
            end
        end

        Node_Loads{idx} = this_node;
        Node_Phases{idx} = this_phases;

    end
    network_json.LVNetwork.Loads = loads_array;



    filename = "network_config.json";

    jsonStr = jsonencode(network_json, 'PrettyPrint', true);
        
    % Write to file
    fid = fopen(fullfile(pwd, filename), 'w');
    if fid == -1
        error('Cannot open file for writing');
    end
    fprintf(fid, '%s', jsonStr);
    fclose(fid);

    save("G_DSS.mat", 'G', 'Node_Phases', 'Node_Loads')

end