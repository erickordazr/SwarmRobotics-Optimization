from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
from pymoo.core.callback import Callback


import math
import random
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

## ********** Auxiliary functions *************

# Function to normalize a value within a range
def normalize(value, min_val, max_val):
    if max_val - min_val == 0:
        return 0
    return (value-min_val) / (max_val-min_val)


# Function to denormalize a value within a range
def denormalize(normalized_value, min_val, max_val):
    return normalized_value * (max_val-min_val) + min_val

def normalize_angle(angle):
    return angle % (2 * math.pi)

def calculate_distance(ind_i, ind_j, delta, distance_range):
    if delta < distance_range / 2:
        distance = np.hypot(ind_i[0] - ind_j[0], ind_i[1] - ind_j[1])
    else:
        distance = math.inf
    return min(distance, math.inf)

# Read database from excel
def DatabaseRead(database):
    # DataBrute=DatabaseRead():
    # Excel reading
    df = pd.read_excel(database)

    Nrows = len(df)
    Ncols = len(df.columns)
    DataBrute = [[0 for i in range(Ncols)] for j in range(Nrows)]
    for r in range(Nrows):
        for c in range(Ncols):
            DataBrute[r][c] = df[df.columns[c]][r]
    return DataBrute


def StoreExcel(Database,write_data):
    # Writing DataBrute in Excel file
    BaseD = Database
    SD = len(Database)
    Index = [j for j in range(SD)]
    columns = ['r_r', 'r_o', 'r_a', 'robots', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6' ]
    df = pd.DataFrame(BaseD, Index, columns)
    print(df)

    # # Create a Pandas Excel writer using XlsxWriter as the engine
    writer = pd.ExcelWriter(write_data, engine='xlsxwriter')
    # # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='Sheet1')
    # # Close the Pandas Excel writer and output the Excel file.
    writer.save()

"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

#  *****     Aggregation task     ******

def dynamic_model(c, t, u):
    # Define parameters of the robot
    params = {
        'm': 0.38,
        'Im': 0.005,
        'd': 0.02,
        'r': 0.03,
        'R': 0.05
    }
    m, Im, d, r, R = params.values()
    
    # Define matrices
    M = np.matrix([[m, 0], [0, Im + m * d ** 2]])
    H = np.array([[-m * d * c[5] ** 2], [m * d * c[4] * c[5]]])
    B = np.matrix([[1 / r, 1 / r], [R / r, -R / r]])
    A = np.matrix([[r / 2, r / 2], [r / (2 * R), -r / (2 * R)]])
    Ts = np.matrix([[0.434, 0], [0, 0.434]])
    Ks = np.matrix([[2.745, 0], [0, 2.745]])
    Kl = np.matrix([[1460.2705, 0], [0, 1460.2705]])

    # Calculate velocity
    dxdt = np.concatenate((
        np.asarray(np.matrix([[np.cos(c[3]), -d * np.sin(c[3])], [np.sin(c[3]), d * np.cos(c[3])]]) * np.array(
            [[c[4]], [c[5]]])),
        np.array([[c[4]], [c[5]]]),
        np.linalg.inv(M + B @ np.linalg.inv(Kl) @ Ts @ np.linalg.inv(A)) @ (B @ np.linalg.inv(Kl) @ Ks @ u - (
                H + B @ np.linalg.inv(Kl) @ np.linalg.inv(A) @ np.array([[c[4]], [c[5]]])))
    ), axis=0)

    return np.squeeze(np.asarray(dxdt))


def movement(ci, u):
    # Define initial and final times
    initial_time, final_time = 0, 1 
    # Generate a sequence of time samples
    t = np.linspace(initial_time, final_time, 10) 
    # Integrate the dynamic model over the given time interval with the given inputs
    c = odeint(dynamic_model, ci, t, args=(u,))

    return c[-1, :]


def foraging(Xs, objects, i_r, animation):
    # optimization_functions = foraging(Xs, objects, animation)
    r_r = Xs[0]
    o_r = Xs[1]
    a_r = Xs[2]
    individuals = int(Xs[3])
    
    cs = np.zeros((individuals, 6))  # Initial individuals states
    c = np.zeros((individuals, 6))  # Individuals states
    report = np.zeros((100000, individuals, 4))  # States report
    state_detected = np.zeros((100000, individuals))
    iterations = 0  # Iterations
    
    # Objective functions
    optimization_functions = np.zeros((6,1))
    f1, f2, f3, f4, f5, f6 = 0, 0, 0, 0, 0, 0
    # execution time, energy used, number of members of the swarm, swarm efficiency, task balance, uncollected objects 
    
    #Parameters enviroment
    wn = random.random() * 0.01  # White noise
    area_limits = 10 # Area limit
    nest_radius = 4  # Maximum distance of influence (nest)
    box_radius = 2.5 # Maximum distance of influence (objects box)
    nestFull = objects
    dirObs = 0
    
    # Visuals
    if animation:
        plt.figure(figsize=(10, 10), dpi=80)
        # ax = plt.gca()  # Nest full (end task)
        
    # Parameters robots
    desired_voltage = np.zeros((individuals, 2))
    repulsion_voltage, orientation_voltage, attraction_voltage, influence_voltage = 2, 2.7, 3.7, 2.7
    repulsion_radius, orientation_radius, attraction_radius = 0.075 + r_r, 0.075 + o_r, 0.075 + a_r

    # Parameters robots - objects
    collectedObjects = np.zeros((individuals, 4))  # Delivery time,Search time,collected objects
    gripState = np.zeros((individuals, 1))  # Open grip/Close grip
    explore = np.zeros((individuals, 1))
    
    # Nest
    nest_arealimits = 0.2
    nest_dot = np.zeros(2) + area_limits * (nest_arealimits / 2)  # Nest dot
    nest_location = [nest_dot[0], nest_dot[1]]  # Nest location (Dotted line)
    nest_influence = np.zeros(individuals)  # Influence of nest activated by individual
    
    # Objects box
    box_center, box_limits = 0.75, 0.2
    objectbox = [box_center * area_limits, box_center * area_limits]
    ob_ip = np.zeros((objects, 2))
    ob_ep = np.zeros((objects, 2))  # Initial position of objects

    objects_location = np.zeros((objects, 2))  # Objects location
    if objects == 0:
        obv = np.zeros((1, 1))  # Objects vector
        goi = np.zeros((1, 1))  # Objects gripped by individual
    else:
        obv = np.zeros((objects, 1))  # Objects vector
        goi = np.zeros((objects, 1))  # Objects gripped by individual

    # Random objects position
    for o in range(objects):
        obRand1 = denormalize(random.random(), box_center - (box_limits / 2), box_center + (box_limits / 2))
        obRand2 = denormalize(random.random(), box_center - (box_limits / 2), box_center + (box_limits / 2))
        objects_location[o] = [area_limits * obRand1, area_limits * obRand2]
        ob_ip[o] = objects_location[o]
        obv[o, 0] = 1 # Search mode

    # initial conditions
    for i in range(individuals):
        if i == 0:
            c[i, :2] = [random.uniform(0, area_limits * 0.25) for _ in range(2)]
        else:
            while True:
                c[i, :2] = [random.uniform(0, area_limits * 0.25) for _ in range(2)]
                if all(math.sqrt((c[i, 0] - c[j, 0]) ** 2 + (c[i, 1] - c[j, 1]) ** 2) > 0.3 for j in range(i)):
                    break
        c[i, 2:] = [0, random.uniform(0, 2 * math.pi), 0, 0]  # Movement, Orientation, Speed, Angular speed
    dirExp = c[:, 3]
               
               
    # Finish task when nest is full
    while nestFull != 0 and iterations<6000:
        iterations = iterations + 1

        for i in range(individuals):
            desired_voltage[i] = [orientation_voltage + wn] * 2

            # Elements detected
            repulsion_walls = np.zeros(individuals)
            repulsion_detected = np.zeros(individuals)
            orientation_detected = np.zeros(individuals)
            attraction_detected = np.zeros(individuals)
            elements_rx, elements_ry = [], []
            elements_ox, elements_oy = [], []
            elements_ax, elements_ay = [], []

            # perception range
            repulsion_range = 6.28319
            orientation_range = 0.5235988
            attraction_range = 0.5235988
            influence_range = 0.5235988
            nest_range = 3.14159
            objectbox_range = 3.14159

            # Verify each sensor for repulsion of walls
            for w in range(5):
                dirObs = c[i, 3] - 3.83972 if w == 0 else dirObs + 1.91986
                dirObs = normalize_angle(dirObs)
                
                Dir = [math.cos(dirObs), math.sin(dirObs)]
                limitX = c[i, 0] + (Dir[0] * repulsion_radius)
                limitY = c[i, 1] + (Dir[1] * repulsion_radius)

                # Resulting direction due exploration
                if limitX > area_limits or limitX < 0 or limitY > area_limits or limitY < 0:
                    dirExp[i] = dirObs + (3 * math.pi / 4) + (random.uniform(0, 1) * math.pi / 2)
                    repulsion_walls[i] += 1
                    repulsion_walls[i] = normalize_angle(repulsion_walls[i])
                    

            # Resulting direction due object box
            objectbox_angle = math.atan2(objectbox[1] - c[i, 1], objectbox[0] - c[i, 0])
            objectbox_angle = normalize_angle(objectbox_angle)

            # Calculation of influence angles by object box
            ob_Beta = objectbox_angle - c[i, 3]
            ob_Beta = normalize_angle(ob_Beta)

            ob_Gamma = c[i, 3] - objectbox_angle
            ob_Gamma = normalize_angle(ob_Gamma)
            
            ob_Delta = min(ob_Beta, ob_Gamma)


            # Calculated distance between the robots and the object zone
            objectbox_distance = calculate_distance(c[i], objectbox, ob_Delta, objectbox_range)


            ob_n = normalize(objectbox_distance, 0, box_radius)
            influence_voltage = denormalize(ob_n, repulsion_voltage, attraction_voltage)

            for j in range(individuals):
                if i == j: # It must not be the same
                    continue
                
                # Angle of the individual with respect to other members of the swarm
                neighbors_angle = math.atan2((c[j, 1] - c[i, 1]), (c[j, 0] - c[i, 0]))
                neighbors_angle = normalize_angle(neighbors_angle)

                # Calculation of angles of repulsion and attraction with respect to other individuals
                beta = neighbors_angle - c[i, 3]
                beta = normalize_angle(beta)
                
                gamma = c[i, 3] - neighbors_angle
                gamma = normalize_angle(gamma)

                delta = min(beta, gamma)

                # Calculation of the repulsion distance with respect to other individuals
                repulsion_distance = calculate_distance(c[i], c[j], delta, repulsion_range)

                # Calculation of the attraction distance with respect to other individuals
                attraction_distance = calculate_distance(c[i], c[j], delta, attraction_range)

                # Calculation of the orientation distance with respect to other individuals
                orientation_distance = calculate_distance(c[i], c[j], delta, orientation_range)
               
                # Count the number of individuals detected in the radius of repulsion, orientation, and attraction
                if repulsion_distance <= repulsion_radius:
                    elements_rx.append(math.cos(neighbors_angle))
                    elements_ry.append(math.sin(neighbors_angle))
                    repulsion_detected[i] += 1

                if orientation_radius < attraction_distance <= attraction_radius and \
                    repulsion_detected[i] == repulsion_walls[i] == 0:
                    elements_ax.append(math.cos(neighbors_angle))
                    elements_ay.append(math.sin(neighbors_angle))
                    attraction_detected[i] += 1

                if repulsion_radius < orientation_distance <= orientation_radius and \
                    repulsion_detected[i] == repulsion_walls[i] == attraction_detected[i] == 0:
                    elements_ox.append(math.cos(c[j, 3]))
                    elements_oy.append(math.sin(c[j, 3]))
                    orientation_detected[i] += 1

            for o in range(objects):
                # Search object
                if obv[o, 0] == 1 and gripState[i, 0] == 0:

                    # Angle of objects
                    object_angle = math.atan2((objects_location[o, 1] - c[i, 1]), (objects_location[o, 0] - c[i, 0]))
                    object_angle = normalize_angle(object_angle)


                    # Calculation of influence angles
                    o_Beta = object_angle - c[i, 3]
                    o_Beta = normalize_angle(o_Beta)

                    o_Gamma = c[i, 3] - object_angle
                    o_Gamma = normalize_angle(o_Gamma)
            
                    o_Delta = min(o_Beta, o_Gamma)


                    # Noise for distance value in influence (standard distribution) of 5%
                    ds1 = sum(random.uniform(-1, 1) for n in range(12))
                    ds2 = sum(random.uniform(-1, 1) for m in range(6))
                    object_noise = (ds1 - ds2) * 0.05


                    # Calculated influence distance
                    object_distance = calculate_distance(c[i], objects_location[o], o_Delta, influence_range)
                    object_distance += object_noise
     
        
                    # Distance between objects and robot
                    object_limit = 0.2
                    if object_distance <= object_limit:
                        nest_influence[i] = 1
                        goi[o] = i
                        collectedObjects[i, 2] += 1
                        gripState[i, 0] = 1  # Close grip
                        obv[o, 0] = 0  # Object taken by robot

                # Nest delivery
                if obv[o, 0] == 0 and nest_influence[i] == 1:

                    # Angle respect to nest
                    nest_angle = math.atan2((nest_location[1] - c[i, 1]), (nest_location[0] - c[i, 0]))
                    nest_angle = normalize_angle(nest_angle)

                    # Calculation of influence angles
                    n_Beta = nest_angle - c[i, 3]
                    n_Beta = normalize_angle(n_Beta)

                    n_Gamma = c[i, 3] - nest_angle
                    n_Gamma = normalize_angle(n_Gamma)

                    n_Delta = min(n_Beta, n_Gamma)
            

                    # Calculated nest distance
                    nest_distance = calculate_distance(c[i], nest_location, n_Delta, nest_range)

                    # Distance between nest and robot
                    nest_limit = 0.2
                    if nest_distance <= nest_limit:
                        objects_location[o] = nest_location
                        nest_dot += np.random.uniform(-0.1, 0.1, size=(2,))
                        nest_location = nest_dot.copy()
                        nestFull -= 1
                        gripState[i, 0] = 0  # Open grip
                        nest_influence[i] = 0  

                    nest_n = normalize(nest_distance, nest_limit, nest_radius)
                    influence_voltage = denormalize(nest_n, repulsion_voltage, attraction_voltage)

                    if gripState[int(goi[o, 0]), 0] == 1:
                        objects_location[o] = c[int(goi[o]), :2]

                ob_ep[o] = objects_location[o]

            # Average of detected elements
            if repulsion_detected[i] > 0:
                repulsion_direction = math.atan2((-np.sum(elements_ry)), (-np.sum(elements_rx)))
                repulsion_direction = normalize_angle(repulsion_direction)

            if orientation_detected[i] > 0:
                orientation_direction = math.atan2((np.sum(elements_oy)), (np.sum(elements_ox)))
                orientation_direction = normalize_angle(orientation_direction)

            if attraction_detected[i] > 0:
                attraction_direction = math.atan2((np.sum(elements_ay)), (np.sum(elements_ax)))
                attraction_direction = normalize_angle(attraction_direction)

            # Behavior Policies
            # Repulsion rules
            if repulsion_walls[i] > 0:
                state_detected[iterations, i] = 1

            if repulsion_detected[i] > 0:
                state_detected[iterations, i] = 1
                if nest_influence[i] == 0:
                    if objectbox_distance < box_radius:
                        explore[i] = 1
                        xT = 0.5 * math.cos(c[i, 3]) + 0.4 * math.cos(repulsion_direction) + 0.1 * math.cos(
                            objectbox_angle)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.4 * math.sin(repulsion_direction) + 0.1 * math.sin(
                            objectbox_angle)
                        # dirExp[i] = c[i,3]  
                    else:
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(repulsion_direction)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(repulsion_direction)
                        dirExp[i] = repulsion_direction
                else:
                    if nest_distance < nest_radius:
                        xT = 0.5 * math.cos(c[i, 3]) + 0.4 * math.cos(repulsion_direction) + 0.1 * math.cos(nest_angle)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.4 * math.sin(repulsion_direction) + 0.1 * math.sin(nest_angle)
                        # dirExp[i] = c[i,3]
                    else:
                        desired_voltage[i, :] = [repulsion_voltage + wn, repulsion_voltage + wn]
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(repulsion_direction)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(repulsion_direction)
                        dirExp[i] = repulsion_direction
                        
                desired_voltage[i, :] = [repulsion_voltage + wn] * 2    
                c[i, 3] = math.atan2(yT, xT)

            # Orientation rules
            if orientation_detected[i] > 0 and repulsion_detected[i] == 0 and attraction_detected[i] == 0 and repulsion_walls[i] == 0:
                if nest_influence[i] == 0:
                    if objectbox_distance < box_radius:
                        state_detected[iterations, i] = 4
                        explore[i] = 1
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(objectbox_angle)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(objectbox_angle)
                        # dirExp[i] = orientation_direction
                    else:
                        state_detected[iterations, i] = 2
                        dirExp[i] = orientation_direction
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(orientation_direction)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(orientation_direction)        
                else:
                    if nest_distance < nest_radius:
                        state_detected[iterations, i] = 4
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(nest_angle)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(nest_angle)
                        # dirExp[i] = orientation_direction
                    else:
                        state_detected[iterations, i] = 2
                        dirExp[i] = orientation_direction
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(orientation_direction)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(orientation_direction) 
                desired_voltage[i, :] = [orientation_voltage + wn] * 2
                c[i, 3] = math.atan2(yT, xT)
                        

            # Attraction rules
            if attraction_detected[i] > 0 and repulsion_detected[i] == 0 and orientation_detected[i] == 0 and repulsion_walls[i] == 0:
                if nest_influence[i] == 0:
                    if objectbox_distance < box_radius:
                        state_detected[iterations, i] = 4
                        explore[i] = 1
                        desired_voltage[i, :] = [orientation_voltage + wn] * 2
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(objectbox_angle)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(objectbox_angle)                       
                        # dirExp[i] = attraction_direction
                    else:
                        state_detected[iterations, i] = 3
                        desired_voltage[i, :] = [attraction_voltage + wn] * 2
                        dirExp[i] = attraction_direction
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(attraction_direction)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(attraction_direction)       
                else:
                    if nest_distance < nest_radius:
                        state_detected[iterations, i] = 4
                        desired_voltage[i, :] = [orientation_voltage + wn] * 2
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(nest_angle)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(nest_angle)
                        # dirExp[i] = attraction_direction
                    else:
                        state_detected[iterations, i] = 3
                        desired_voltage[i, :] = [attraction_voltage + wn] * 2
                        dirExp[i] = attraction_direction
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(attraction_direction)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(attraction_direction)
                c[i, 3] = math.atan2(yT, xT)
                        

            # Orientation and Attraction rules
            if orientation_detected[i] > 0 and attraction_detected[i] > 0 and repulsion_detected[i] == 0 and repulsion_walls[i] == 0:
                if nest_influence[i] == 0:
                    if objectbox_distance < box_radius:
                        state_detected[iterations, i] = 4
                        explore[i] = 1
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(objectbox_angle)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(objectbox_angle)                       
                        # dirExp[i] = attraction_direction
                    else:
                        state_detected[iterations, i] = 5
                        dirExp[i] = math.atan2((math.sin(orientation_direction) + math.sin(attraction_direction)),
                                                (math.cos(orientation_direction) + math.cos(attraction_direction)))
                        xT = 0.5 * math.cos(c[i, 3]) + 0.25 * math.cos(orientation_direction) + 0.25 * math.cos(
                            attraction_direction)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.25 * math.sin(orientation_direction) + 0.25 * math.sin(
                            attraction_direction)     
                else:
                    if nest_distance < nest_radius:
                        state_detected[iterations, i] = 4
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(nest_angle)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(nest_angle)
                        # dirExp[i] = attraction_direction
                    else:
                        state_detected[iterations, i] = 5
                        dirExp[i] = math.atan2((math.sin(orientation_direction) + math.sin(attraction_direction)),
                                                (math.cos(orientation_direction) + math.cos(attraction_direction)))
                        xT = 0.5 * math.cos(c[i, 3]) + 0.25 * math.cos(orientation_direction) + 0.25 * math.cos(
                            attraction_direction)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.25 * math.sin(orientation_direction) + 0.25 * math.sin(
                            attraction_direction)       
                desired_voltage[i, :] = [orientation_voltage + wn] * 2
                c[i, 3] = math.atan2(yT, xT)
            
                        
            # Out of range
            if attraction_detected[i] == 0 and repulsion_detected[i] == 0 and orientation_detected[i] == 0 and repulsion_walls[i] == 0:
                if nest_influence[i] == 0:
                    if objectbox_distance < box_radius:
                        state_detected[iterations, i] = 4
                        explore[i] = 1
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(objectbox_angle)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(objectbox_angle)                      
                    else:
                        state_detected[iterations, i] = 0
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(dirExp[i])
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(dirExp[i])
                else:
                    if nest_distance < nest_radius:
                        state_detected[iterations, i] = 4
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(nest_angle)
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(nest_angle)
                    else:
                        state_detected[iterations, i] = 0
                        xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(dirExp[i])
                        yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(dirExp[i])
                        
                desired_voltage[i, :] = [orientation_voltage + wn] * 2
                c[i, 3] = math.atan2(yT, xT)
                

            if explore[i] == 1 and objectbox_distance > box_radius and random.random() < 0.1:
                explore[i] = 0
                dirExp[i] = dirExp[i] + (3 * math.pi / 4) + (random.random() * math.pi / 2)
                dirExp[i] = normalize_angle(dirExp[i])

            report[iterations, i, 0] = c[i, 0]
            report[iterations, i, 1] = c[i, 1]
            report[iterations, i, 2] = c[i, 2]
            report[iterations, i, 3] = c[i, 3]
            collectedObjects[i, 3] = c[i, 2]

            c_past = c[i, :]
            cs[i, :] = movement(c[i, :], desired_voltage[i, :].reshape(2, 1))
            c[i, :] = cs[i, :]

            if not (0 + repulsion_radius < c[i, 0] < area_limits - repulsion_radius and
                    0 + repulsion_radius < c[i, 1] < area_limits - repulsion_radius):
                c[i, :] = c_past

            # this avoids an infinite increment of radians
            c[i, 3] = normalize_angle(c[i, 3])

            # Delivery time
            if gripState[i, 0] == 1:  # Grip close
                collectedObjects[i, 0] = collectedObjects[i, 0] + 1

        # Simulation
        if animation:

            # plt.figure(figsize=(7, 7), dpi=80)
            ax = plt.gca()
            x = report[iterations, :, 0]
            y = report[iterations, :, 1]
            vx = np.cos(report[iterations, :, 3])
            vy = np.sin(report[iterations, :, 3])
            
            colors = ['dimgray' if state_detected[iterations, i] == 0 else 
                      'red' if state_detected[iterations, i] == 1 else 
                      'blue' if state_detected[iterations, i] == 2 else 
                      'lime' if state_detected[iterations, i] == 3 else 'yellow' for i in range(individuals)]
            
            box_circle = plt.Circle((objectbox[0], objectbox[1]), box_radius, color='blue', alpha=0.6, fill=False)
            box_rectangle = plt.Rectangle(
                ((box_center - (box_limits / 2)) * area_limits, (box_center - (box_limits / 2)) * area_limits),
                box_limits * area_limits, box_limits * area_limits, color='blue', alpha=0.6, fill=False)
            
            nest_circle = plt.Circle((nest_location[0], nest_location[1]), nest_radius, color='red', alpha=0.6,
                                        fill=False)
            nest_rectangle = plt.Rectangle((0, 0), nest_arealimits * area_limits, nest_arealimits * area_limits,
                                            color='red',
                                            alpha=0.6, fill=False)

            plt.cla()
            ax.add_patch(box_circle)
            ax.add_patch(box_rectangle)
            ax.add_patch(nest_circle)
            ax.add_patch(nest_rectangle)
            
            ax.quiver(x, y, vx, vy, color=colors)
            for o in range(objects):
                obplt = plt.Circle((objects_location[o, 0], objects_location[o, 1]), 0.1, color='#00BB2D')
                ax.add_patch(obplt)

            ax.set(xlim=(0, area_limits), ylim=(0, area_limits), aspect='equal')
            plt.pause(0.000001)

    collectedObjects[:, 1] = iterations - collectedObjects[:, 0]
    np.save('report', report)
    np.save('collectedObjects', collectedObjects)
    
    f1 = iterations
    f2 = sum(collectedObjects[:, 3])
    f3 = individuals
    f4 = sum(collectedObjects[:, 1]) / (iterations*individuals)
    f5 = np.std(collectedObjects[:, 2])
    f6 = nestFull
    optimization_functions = np.array([f1, f2, f3, f4, f5, f6]).reshape(6, 1)

    return optimization_functions
"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

def simulation_mean(X, replicas, objects, i_r):
    # (out, R, table) = simulation_mean(X, replicas, objects, i_r)

    optimization_functions_report = np.zeros((replicas, 6))
    optimization_functions_replicas = np.zeros((replicas, 6))
    R = np.zeros((6))
    animation = False
    Xs = np.zeros(len(X[:]))
    f1 = np.zeros(len(X[:]))
    f2 = np.zeros(len(X[:]))
    f3 = np.zeros(len(X[:]))
    f4 = np.zeros(len(X[:]))
    f5 = np.zeros(len(X[:]))
    f6 = np.zeros(len(X[:]))
    
    for x in range(len(X[:])):
        Xs = np.zeros(4)
        
        Xs[0] = X[x][0]
        Xs[1] = X[x][1]
        Xs[2] = X[x][2]
        Xs[3] = int(X[x][3])

        for r in range(replicas):
            optimization_functions = foraging(Xs, objects, i_r, animation)
            # print("Progress: ", round(percentage[r + 1], 2), "%")

            for f in range(6):
                optimization_functions_report[r, f] = optimization_functions[f, 0]

        for r in range(replicas):
            if r == 0:
                optimization_functions_replicas[r] = optimization_functions_report[r, :]
            elif r > 0:
                previous_mean = (optimization_functions_replicas[0:r, :].sum(axis=0) + optimization_functions_report[r, :]) / (r + 1)
                optimization_functions_replicas[r] = previous_mean
        
        for i in range(6):
            R[i] = optimization_functions_replicas[replicas - 1, i]
    
        f1[x] = R[0]
        f2[x] = R[1]
        f3[x] = R[2]
        f4[x] = R[3]  
        f5[x] = R[2]
        f6[x] = R[3]  
    
    return np.column_stack([f1, f2, f3, f4, f5, f6])

"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

class ProgressCallback(Callback):
    def __init__(self, n_gen):
        super().__init__()
        self.n_gen = n_gen

    def __call__(self, problem, algorithm, **kwargs):
        gen = algorithm.n_gen
        progress = 100 * gen / self.n_gen
        print(f"Progreso: {progress:.2f}%")
        
"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"


# Define a problem class that inherits from Problem
class foragingTask(Problem):

    # Constructor method with the number of variables n_var
    def __init__(self, n_var=4, **kwargs):
        # Call the super class constructor to initialize the problem
        super().__init__(n_var=n_var, n_obj=6, n_ieq_constr=0, vtype=float, **kwargs)

    # Method to evaluate the problem
    def _evaluate(self, x, out, *args, **kwargs):
        # Fill out with the objectives calculated in the `simulation_mean` function
        replicas = 10
        objects = 20
        i_r = 3
        out["F"]  = simulation_mean(x, replicas, objects, i_r)
        
# Create an instance of the foraging problem with 4 variables 
# repulsion_radius, orientation_radius, attraction_radius, individuals
xl = [0.1, 0.2, 0.4, 5]
xu = [0.4, 1.2, 3,   20]
problem = foragingTask(n_var=4, xl=xl, xu=xu)


# Get the reference directions for the problem
ref_dirs = get_reference_directions("das-dennis", 6, n_partitions=3)

# Create an instance of the NSGA3 algorithm with the reference directions
algorithm = NSGA3(pop_size=100, ref_dirs=ref_dirs)

# Minimize the problem with the NSGA3 algorithm, 20 generations, seed=1 and verbose=False
n_gen = 4
callback = ProgressCallback(n_gen)
res = minimize(problem, algorithm, ('n_gen', n_gen), seed=1, callback=callback, verbose=False)

# Print the final result
print(res.F)

# Visualize the result
Scatter().add(res.F).show()