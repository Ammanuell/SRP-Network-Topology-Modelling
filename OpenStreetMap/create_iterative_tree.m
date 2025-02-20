function [G, pole_positions] = create_iterative_tree(houses_longlat, pole_positions, house_voltages)
    % Not finished
    % Create edges
    edges = [];
    weights = [];
    
    % Remove poles not closest to a house
    n = length(pole_positions);
    poles_connected = zeros(1, n);
    for i = 1:length(houses_longlat)
        % Find closest pole
        distances = [];
        for j = 1:length(pole_positions)
            dist = abs(haversine(houses_longlat(i,:), pole_positions(j,:)));
            distances = [distances; dist];
        end
       
        [min_dist, closest_pole] = min(distances);
    
        poles_connected(closest_pole) = 1;
        
    end
    pole_positions = pole_positions(poles_connected~=0, :);

    % Connect each customer to closest pole
    n = length(houses_longlat);
    for i = 1:length(houses_longlat)
        % Find closest pole
        distances = [];
        for j = 1:length(pole_positions)
            dist = abs(haversine(houses_longlat(i,:), pole_positions(j,:)));
            distances = [distances; dist];
        end
       
        [min_dist, closest_pole] = min(distances);
    
        % Create edge
        edges = [edges; [i, closest_pole+n]];
        weights = [weights; min_dist];
        
    end
    
    G = graph(edges(:,1), edges(:,2), weights);

    figure
    p = plot(G);
    p.XData = [houses_longlat(:,1); pole_positions(:,1)];
    p.YData = [houses_longlat(:,2); pole_positions(:,2)];

    % Calculate 'voltage' of each pole
    % pole_voltages = zeros(length(pole_positions), length(house_voltages(1,:)));
    pole_voltages = zeros(length(pole_positions), 2000, 3);

    % For all poles connected to houses, voltage is average of them
    for i = n+1:n+length(pole_positions)
        houses = neighbors(G, i);
        %houses = houses(houses>length(pole_positions)); % Get all houses connected to pole
        
        if ~isempty(houses)
            pole_voltages(i-n,:) = mean(house_voltages(houses, :),1, 'omitnan');
        end
    end
    
    x = 0;
    new_edges = [];
    while x < 1
        % Calculate correlations
        corr_table_A = corrcoef(pole_voltages(:,:,1)');
        corr_table_B = corrcoef(pole_voltages(:,:,2)');
        corr_table_C = corrcoef(pole_voltages(:,:,3)');
    
        % Remove unneeded corrrelations
        corr_table_A(corr_table_A == 1) = NaN;
        corr_table_B(corr_table_B == 1) = NaN;
        corr_table_C(corr_table_C == 1) = NaN;
    
        for i = 1:length(new_edges)
            edge = new_edges(i,:);
            corr_table_A(edge(1), edge(2)) = NaN;
            corr_table_A(edge(2), edge(1)) = NaN;
            corr_table_B(edge(1), edge(2)) = NaN;
            corr_table_B(edge(2), edge(1)) = NaN;
            corr_table_C(edge(1), edge(2)) = NaN;
            corr_table_C(edge(2), edge(1)) = NaN;
        end

        % Add best edge
        [max_A, row_max_A] = max(max(corr_table_A, [], 1)); 
        [~, col_max_A] = max(corr_table_A(:, row_max_A)); 

        [max_B, row_max_B] = max(max(corr_table_B, [], 1)); 
        [~, col_max_B] = max(corr_table_B(:, row_max_B)); 

        [max_C, row_max_C] = max(max(corr_table_C, [], 1)); 
        [~, col_max_C] = max(corr_table_C(:, row_max_C)); 

        maxes = [max_A, max_B, max_C];
        rows = [row_max_A, row_max_B, row_max_C];
        cols = [col_max_A, col_max_B, col_max_C];

        [~, index] = max(maxes);
        pole_1 = rows(index) + n;
        pole_2 = cols(index) + n;

        edge = [pole_1, pole_2];

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