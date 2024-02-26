file_path = 'C:\Users\Lenovo\Desktop\研究生\TSP Solution\dataset\pbn423.tsp';

data = dlmread(file_path, ' ', [8 1 430 2]);

% figure('Units', 'inches', 'Position', [1, 1, 8, 8]);

scatter(data(:,1), data(:,2), 45, 'k.', 'MarkerEdgeAlpha', 0.6);

xlim([-10, max(data(:,1)) + 10]);
ylim([-10, max(data(:,2)) + 10]);

% xlabel('X-axis');
% ylabel('Y-axis');
% title('Scatter plot of TSP for 423 nodes');

% set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [0, 0, 8, 8]);