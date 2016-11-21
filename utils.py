import math;
import random;

SEED = 5;

#Path loss parameters
POWER_T = 10 * math.log10(16 * 0.001);# Transmitted power mW to dBm 
ALPHA = 2.5;

# configuration parameters
K = 40;
T = 5;
SD = 2;
WIDTH = 200;
FEATURES_PER_CELL = 200;

#get actual X and Y co-ordinates given grid indices
def get_centroid(grid_index, K):
	return ((grid_index * K) + (K / 2));

def generate_transmitter_locations(K, T, seed):
	print("Generating Transmitter Locations.....");
	random.seed(seed);
	cells_per_row = int(WIDTH / K);
	cells_per_col = cells_per_row;

	# generating random grid indices
	grid_row = random.sample(range(cells_per_row), T); 
	grid_col = random.sample(range(cells_per_col), T);
	grid_loc_transmitters = [(grid_row[i], grid_col[i]) for i in range(len(grid_row))];

	#computing actual x, y cordinates
	centroid_x = [get_centroid(row_index, K) for row_index in grid_row ]; 
	centroid_y = [get_centroid(col_index, K) for col_index in grid_col ];
	transmitters = [(centroid_x[i], centroid_y[i]) for i in range(len(centroid_x))];

	return grid_loc_transmitters, transmitters;

def eucledian_distance(v1,v2):
	if len(v1) != len(v2):
		print("***Both vectors are not of equal length!!***");
	square_differences = [(v1[i] - v2[i]) ** 2 for i in range(len(v1))];
	return math.sqrt(sum(square_differences));

def generate_power_at_d(d, K, SD, seed):
	#random.seed(seed);
	noise = random.gauss(0, SD);
	pl = 10 * ALPHA * math.log10(d) + noise;
	pr = POWER_T - pl;		
	return pr;

	