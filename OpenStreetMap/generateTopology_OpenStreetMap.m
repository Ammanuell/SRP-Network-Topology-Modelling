close all; clc; clear;

% Overpass API
url = 'https://overpass-api.de/api/interpreter';
   

% Bounding Box
east = 144.698578;
north = -37.837436;
west = 144.695397;
south = -37.838963;
bbox = "(" + south + "," + west + "," + north + "," + east +  ")";

% Choose filter for houses - Depending on the area, different filters must be used
house_filter = 'node["addr:housenumber"]'; % Houses are only simple nodes

% Road filter 
roads_filter = 'way[highway~"motorway|trunk|primary|secondary|tertiary|residential|service|unclassified|road"]';

house_query = "[out:json];" + house_filter + bbox +' ;out;';
roads_query = "[out:json];" + roads_filter + bbox +' ;out geom;';

% Send the request
house_data = webwrite(url, 'data', house_query);
house_data = house_data.elements;

roads_data = webwrite(url, 'data', roads_query);
roads_data = roads_data.elements;

houses_longlat = [vertcat(house_data.lon), vertcat(house_data.lat)];

%% Plot OpenStreetMap Data returned
% Plot houses
figure;
p1 = plot(houses_longlat(:,1), houses_longlat(:,2), 'square');
axis equal
hold on;

% Plot bounding box
p2 = plot(linspace(east,west,1000),south*ones(1,1000), '--', 'Color', 'magenta');
plot(linspace(east,west,1000),north*ones(1,1000), '--', 'Color', 'magenta');
plot(east*ones(1,1000), linspace(south,north,1000), '--', 'Color', 'magenta');
plot(west*ones(1,1000), linspace(south,north,1000), '--', 'Color', 'magenta');

% Plot roads
for i = 1:length(roads_data)
    coords = roads_data(i).geometry;
    lats = [coords.lat];
    lons = [coords.lon];
    
    % Plot original road
    p3 = plot(lons, lats, '-', 'Color', [0.5, 0.5, 0.5]);  
    %plot(lons, lats, 'o', 'Color', [1 0.5 0]);        % Original nodes

end


% Generate pole positions, we are assuming we know the pole positions

pole_positions = generate_pole_positions(roads_data, 0);

% Plot pole_positions
p4 = plot(pole_positions(:,1), pole_positions(:,2), '.', 'Color', 'r'); 

legend([p1 p2 p3 p4], {'houses', 'bounding box', 'roads', 'pole positions'});


%% Ground truth - Generate random likely topology

% Simple distance mst with 0.3 variability in distances
[G_groundtruth, pole_positions_groundtruth] = create_distance_MST(houses_longlat, pole_positions, 0.3); 
[G_groundtruth, pole_positions_groundtruth] = prune_tree(G_groundtruth, pole_positions_groundtruth);

figure
p = plot(G_groundtruth);
p.XData = [pole_positions_groundtruth(:,1);  houses_longlat(:,1)];
p.YData = [pole_positions_groundtruth(:,2); houses_longlat(:,2)];
p.NodeLabel = {};
title('Ground Truth Model')


%% Run Load Flow to generate customer voltages

run_load_flow(G_groundtruth, pole_positions_groundtruth, houses_longlat);

customer_data = load('Synthetic_data.mat');

% Create voltage table
house_voltages_1p = zeros(length(houses_longlat), 2000); % Combine all phases into one
house_voltages_3p = zeros(length(houses_longlat), 2000, 3); % Keep phases seperate
fields = {'V1', 'V2', 'V3'};
for i = 1:length(customer_data.scriptvar.ABC_Nodes)
    
    ids = customer_data.scriptvar.ABC_Nodes{i};
    house_voltages_1p(ids - length(pole_positions_groundtruth), :) = customer_data.scriptvar.(fields{i});
    house_voltages_3p(ids - length(pole_positions_groundtruth), :, i) = customer_data.scriptvar.(fields{i});
end

house_voltages_3p(house_voltages_3p == 0) = NaN; % Make 0 values (not known) NaN so that it does not affect averaging later


%% Recreate topology using GIS coordinates of poles and houses, plus voltages of houses

% Estimation
% Created with minimum spanning tree of houses and 

[G_estimate, pole_positions_estimate] = create_distance_MST(houses_longlat, pole_positions_groundtruth, 0);

figure
p = plot(G_estimate);
p.XData = [pole_positions_estimate(:,1);  houses_longlat(:,1)];
p.YData = [pole_positions_estimate(:,2); houses_longlat(:,2)];
p.NodeLabel = {};
title('Simple Distance MST')

[G_estimate, pole_positions_estimate] = create_weighted_MST_3p(houses_longlat, pole_positions_groundtruth, house_voltages_3p);
% [G_estimate, pole_positions_estimate] = create_weighted_MST_1p(houses_longlat, pole_positions_groundtruth, house_voltages_1p);


figure
p = plot(G_estimate);
p.XData = [houses_longlat(:,1); pole_positions_estimate(:,1)];
p.YData = [houses_longlat(:,2); pole_positions_estimate(:,2)];
p.NodeLabel = {};
title('Weighted MST')


%%
function run_load_flow(G, pole_positions, houses_longlat)
    
    % Prune edges/nodes that don't lead to a customer
    [G, pole_positions] = prune_tree(G, pole_positions);
    
    % Add phases
    G = add_phases(G);

    % Add transformer 
    % G = add_transformer(G, pole_positions, houses_longlat);


        
    create_network_json(G);

    pyenv('Version', 'C:\Users\Amman\.pyenv\pyenv-win\versions\3.9.13\python.exe');
    
    % n_times controls how many samples we generate
    % tr_model controls what the transformer looks like (0 for constant, 1 for
    %   sinusoidal, 2 for synthetic (unavalible))
    % load_profile_model controls what the loads look like (0 for synthetic, 1
    %   for real (unavalible)
    % pv_pen controls the pv penetration (usually ~30%)
    pyrunfile("3_p_modelling_Python.py", "", n_times=2000, tr_model=1, load_profile_model = 0, pv_pen=0.3)
                
    % Standard deviation of voltage error (0.5v)
    Compare_Load_Flow(0.5)
    load('Synthetic_data.mat')
    
    figure()
    subplot(2, 1, 1)
    heatmap(scriptvar.ABC_Nodes{1}, scriptvar.ABC_Nodes{1}, distances(G, scriptvar.ABC_Nodes{1}, scriptvar.ABC_Nodes{1}))
    title("Distances between nodes in Phase A")
    subplot(2, 1, 2)
    heatmap(scriptvar.ABC_Nodes{1}, scriptvar.ABC_Nodes{1}, corrcoef(scriptvar.V1'))
    title("Correlation between voltages in Phase A")

    figure()
    scatter(distances(G, scriptvar.ABC_Nodes{1}, scriptvar.ABC_Nodes{1}), corrcoef(scriptvar.V1'))
end


function [G, pole_positions] = prune_tree(G, pole_positions)
    % Prune edges/nodes that don't lead to a customer
    stop = 0;
   
    % Iteravely remove all leaf nodes that are poles
    while ~stop
        stop = 1;
        rem =  zeros(length(pole_positions), 1);
        for i = 1:length(pole_positions)
            if degree(G, i) == 1
                rem(i) = 1;
                G = rmnode(G, i);
                stop = 0;
                break;
            end
        end
        pole_positions = pole_positions(logical(~rem), :);

    end
end

function G = add_transformer(G, pole_positions, houses_longlat)
    % Add transformer 
    transformer_position = sum(centrality(G,'closeness', 'Cost',G.Edges.Weight) .* [pole_positions;  houses_longlat]) * 1/sum(centrality(G,'closeness', 'Cost',G.Edges.Weight));
    
    % Find closest pole and connect transformer to it
    closest_pole = 1;
    closest_pole_dist = abs(haversine(transformer_position, pole_positions(1,:)));

    for i = 2:length(pole_positions)
        dist = abs(haversine(transformer_position, pole_positions(i,:)));
        if dist < closest_pole_dist
                closest_pole = i;
                closest_pole_dist = dist;
        end
    end

    edges = G.Edges.EndNodes;
    weights = G.Edges.Weight;

    % Add edge to transformer
    G = graph(edges(:,1)+1, edges(:,2)+1, weights);
    edge = table([closest_pole + 1, 1] , closest_pole_dist, 1, 1, 1 , ...
        'VariableNames',{'EndNodes','Weight','pA', 'pB', 'pC'});
    edge = [closest_pole + 1, 1];
    G = addedge(G, edge(1), edge(2), closest_pole_dist); % Transformer added

    
    
    % figure
    % p = plot(G);
    % p.XData = [transformer_position(1) ; pole_positions(:,1);  houses_longlat(:,1)];
    % p.YData = [transformer_position(2) ; pole_positions(:,2); houses_longlat(:,2)];
    % p.NodeLabel = {};
    % title('ground truth with transformer')
   
end

function G = add_phases(G)

    leaf_nodes = find(degree(G)==1);
    leaf_nodes = leaf_nodes(leaf_nodes~=1);
    
    for i=1:length(leaf_nodes)
    
        % Allocate this to either Phase A, Phase B or Phase C
    
        ph = randi([1, 3]);

        G.Nodes.Phase(leaf_nodes(i)) = ph;


    
    end

    for i = 1:height(G.Edges.EndNodes)
    
        n1 = G.Edges.EndNodes(i, 1);
        n2 = G.Edges.EndNodes(i, 2);
    
        if G.Nodes.Phase(n1)==0 && G.Nodes.Phase(n2)==0
            G.Edges.pA(i) = 1;
            G.Edges.pB(i) = 1;
            G.Edges.pC(i) = 1;
        end
    
        if G.Nodes.Phase(n1)==1 || G.Nodes.Phase(n2)==1
            G.Edges.pA(i) = 1;
        end
    
        if G.Nodes.Phase(n1)==2 || G.Nodes.Phase(n2)==2
            G.Edges.pB(i) = 1;
        end
    
        if G.Nodes.Phase(n1)==3 || G.Nodes.Phase(n2)==3
            G.Edges.pC(i) = 1;
        end

    end



end

function pole_positions = generate_pole_positions(roads_data, randomness)
    pole_positions = [];
    pole_positions_xy = [];
    spacing_og = 40;
    min_dist = inf;

    lower_bound = 1 - randomness;
    a = lower_bound;
    b = 2*(1-lower_bound);
    
    for i = 1:length(roads_data)
        coords = roads_data(i).geometry;
        lats = [coords.lat];
        lons = [coords.lon];
        
        % Convert lat/lon to x/y 
        [x, y] = ll2meters(lons, lats);
        coords_xy = [x',y'];
    
        rng = a + b*rand();
        r_dist = abs((1-rng)*spacing_og);

        for j= 1:length(coords_xy)-1
            dist = sqrt((x(j+1)-x(j))^2 +(y(j+1)-y(j))^2);  
            % If distance betwen road corners not long enough, skip past
            if dist < r_dist
                r_dist = r_dist - dist;
            else
                % If long enough, pole_positions in between road corners
                direction = coords_xy(j+1,:) - coords_xy(j,:);
                direction_vec = direction/norm(direction);
                pole_pos_xy = coords_xy(j,:) + r_dist*direction_vec;
    
                while r_dist < dist
                    % Check if pole too close to other pole_positions
                    if ~isempty(pole_positions_xy)
                        distances = vecnorm(pole_positions_xy - pole_pos_xy, 2, 2); 
                        min_dist = min(distances);
                    end
                    % If not too close, add as pole
                    if min_dist > 15
                        [pole_pos_lon, pole_pos_lat] = meters2ll(pole_pos_xy(1), pole_pos_xy(2));
        
                        pole_pos_lonlat = [pole_pos_lon', pole_pos_lat'];
                        pole_positions = [pole_positions;pole_pos_lonlat];
                        pole_positions_xy = [pole_positions_xy;pole_pos_xy];
                    end
                    
                    % Calculate next pole position
                    rng = a + b*rand();
                    spacing = rng*spacing_og;

                    r_dist = r_dist + spacing;
                    pole_pos_xy = coords_xy(j,:) + r_dist*direction_vec;


                end
                r_dist = r_dist-dist;
            end
        end
            
    end
end
function d = haversine(point_1, point_2)
    R = 6378137;
    
    lon1 = deg2rad(point_1(1));
    lat1 = deg2rad(point_1(2));

    lon2 = deg2rad(point_2(1));
    lat2 = deg2rad(point_2(2));

    delta_lat = lat2 - lat1;
    delta_lon = lon2 - lon1;

    % Haversine formula
    a = sin(delta_lat/2)^2 + cos(lat1) * cos(lat2) * sin(delta_lon/2)^2;
    c = 2 * atan2(sqrt(a), sqrt(1 - a));

    % Compute distance
    d = R * c/1000;
end


function [x, y] = ll2meters(lon, lat)
    R = 6378137; % Earth radius in meters
    x = R * deg2rad(lon);
    y = R * log(tan(pi/4 + deg2rad(lat)/2));
end

function [lon, lat] = meters2ll(x, y)
    R = 6378137; % Earth radius in meters
    lon = rad2deg(x / R);
    lat = rad2deg(2 * atan(exp(y / R)) - pi/2);
end