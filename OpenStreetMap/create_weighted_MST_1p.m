function [G, pole_positions] = create_weighted_MST_1p(houses_longlat, pole_positions, house_voltages)

    
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
    n = length(pole_positions);
    for i = 1:length(houses_longlat)
        % Find closest pole
        distances = [];
        for j = 1:length(pole_positions)
            dist = abs(haversine(houses_longlat(i,:), pole_positions(j,:)));
            distances = [distances; dist];
        end
       
        [min_dist, closest_pole] = min(distances);
    
        % Create edge
        edges = [edges; [n+i, closest_pole]];
        weights = [weights; min_dist];
        
    end
    
    G_temp = graph(edges(:,1), edges(:,2), weights);

    % Calculate 'voltage' of each pole
    % pole_voltages = zeros(length(pole_positions), length(house_voltages(1,:)));
    pole_voltages = zeros(length(pole_positions), 2000);

    % For all poles connected to houses, voltage is average of them
    for i = 1:length(pole_positions)
        houses = neighbors(G_temp, i);
        houses = houses(houses>length(pole_positions)); % Get all houses connected to pole
        
        if ~isempty(houses)
            pole_voltages(i,:) = mean(house_voltages(houses - length(pole_positions), :),1);
        end
    end

    % Create edges between pole, weighted based on voltage diff, correlation and distance
    start = length(edges);
    edges = [edges ; nchoosek(1:length(pole_positions), 2)];
    
    corr_table = corrcoef(pole_voltages');
    
    
    for i = start+1:length(edges)
        edge = edges(i,:);
        
        pole_1 = pole_positions(edge(1),:);
        pole_2 = pole_positions(edge(2),:);
        
        dist = abs(haversine(pole_1, pole_2));

        voltage_diff = abs(pole_voltages(edge(1)) - pole_voltages(edge(2)));

        corr =  corr_table(edge(1), edge(2));
        
        % weight = dist + 0.01*voltage_diff;
        weight = dist - 0.1*corr;
        
        weights = [weights; weight];
    end
    
    
    % Create graph and MST
    G = graph(edges(:,1), edges(:,2), weights);
    G = minspantree(G);

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

