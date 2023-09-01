from pymoo.core.problem import Problem
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

import math
import random
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

## ********** Auxiliary functions *************

# Function to normalize a value within a range
def normalize(value, min_val, max_val):
    # Check if the range is zero
    if max_val - min_val == 0:
        # Return 0 if the range is zero to avoid division by zero error
        return 0
    # Normalize the value by dividing it by the range and subtracting the minimum value
    return (value-min_val) / (max_val-min_val)

# Function to denormalize a value within a range
def denormalize(normalized_value, min_val, max_val):
    # Denormalize the value by multiplying the normalized value by the range and adding the minimum value
    return normalized_value * (max_val-min_val) + min_val

"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

#  *****     Aggregation task     ******

def model(c, t, u):
    # Robot parameters
    m = 0.38  # Robot mass (kg)
    Im = 0.005  # Moment of inertia
    d = 0.02  # Distance from centroid to wheel axis (m)
    r = 0.03  # Wheel radius (m)
    R = 0.05  # Distance to wheel-center (m)

    M = np.matrix([[m, 0], [0, Im + m * d ** 2]])  # Inertia matrix
    H = np.array([[-m * d * c[5] ** 2], [m * d * c[4] * c[5]]])  # Coriolis matrix
    B = np.matrix([[1 / r, 1 / r], [R / r, -R / r]])  # Conversion matrix torque-wheel-mobile force
    A = np.matrix([[r / 2, r / 2], [r / (2 * R), -r / (2 * R)]])
    Ts = np.matrix([[0.434, 0], [0, 0.434]])
    Ks = np.matrix([[2.745, 0], [0, 2.745]])
    Kl = np.matrix([[1460.2705, 0], [0, 1460.2705]])

    dx_dt = np.array(np.concatenate((
        np.asarray(np.matrix([[np.cos(c[3]), -d * np.sin(c[3])], [np.sin(c[3]), d * np.cos(c[3])]]) * np.array(
            [[c[4]], [c[5]]])),
        np.array([[c[4]], [c[5]]]),
        np.linalg.inv(M + B * np.linalg.inv(Kl) * Ts * np.linalg.inv(A)) * (B * np.linalg.inv(Kl) * Ks * u - (
                H + B * np.linalg.inv(Kl) * np.linalg.inv(A) * np.array([[c[4]], [c[5]]])))
    ), axis=0))

    dx_dt.reshape(6, 1)
    return np.squeeze(np.asarray(dx_dt))


def movement(ci, u):
    initial_time = 0
    final_time = 1
    t = np.linspace(initial_time, final_time, 10)
    c = odeint(model, ci, t, args=(u,))

    rows = len(c)
    c_ = c[rows - 1, :]

    return c_


def aggregation(Xs, iterations, i_r, animation):
    r_r = Xs[0]
    o_r = Xs[1]
    a_r = Xs[2]
    individuals = int(Xs[3])
    cs = np.zeros((individuals, 6))  # Initial individuals states
    c = np.zeros((individuals, 6))  # Individuals states
    report = np.zeros((iterations, individuals, 8))  # States report
    wn = random.random() * 0.01  # White noise
    area_limits = 10  # Area limit
    influence_position = [area_limits * 0.75, area_limits * 0.75]  # Influence position
    state_detected = np.zeros((iterations, individuals))

    # Optimization
    optimization_functions = np.zeros((4, 1))
    localized_influence = np.zeros((individuals, 1))  # location time by individual
    localized_influence_flag = 0
    cm_ = np.zeros((iterations, 5))
    failure_radio = np.zeros((iterations, individuals))
    amount_f3 = np.zeros(iterations)
    f1 = iterations  # Location time for the swarm
    f2 = 0  # Average distance to center of mass
    f3 = 0  # radio of failure(every member of the swarm locate objective)
    f4 = individuals  # number of members of the swarm

    dirObs = 0
    repulsion_direction = 0
    orientation_direction = 0
    attraction_direction = 0
    influence_direction = 0

    if animation:
        plt.figure(figsize=(7, 7), dpi=80)
        ax = plt.gca()

    xn = np.zeros((3, individuals))
    yn = np.zeros((3, individuals))
    vxn = np.zeros((3, individuals))
    vyn = np.zeros((3, individuals))
    data = np.zeros((3, 5))
    index_n = round(iterations / 2) + 1

    desired_voltage = np.zeros((individuals, 2))
    repulsion_voltage = 2  # 15 cm/s
    orientation_voltage = 2.7  # 20 cm/s
    attraction_voltage = 3.7  # 30 cm/s
    influence_voltage = orientation_voltage

    repulsion_radius = 0.075 + r_r
    orientation_radius = 0.075 + o_r
    attraction_radius = 0.075 + a_r

    # initial conditions
    for i in range(individuals):
        if i == 0:
            c[i, 0] = random.uniform(0, area_limits * 0.25)  # x position
            c[i, 1] = random.uniform(0, area_limits * 0.25)  # y position
        else:
            while True:
                safe = 0
                c[i, 0] = random.uniform(0, area_limits * 0.25)  # x position
                c[i, 1] = random.uniform(0, area_limits * 0.25)  # y position
                for j in range(i):
                    d_other = math.sqrt((c[i, 0] - c[j, 0]) ** 2 + (c[i, 1] - c[j, 1]) ** 2)
                    if d_other > 0.3:
                        safe = safe + 1
                if safe == i:
                    break
        c[i, 2] = 0  # Movement
        c[i, 3] = random.uniform(0, 2 * math.pi)  # Orientation
        c[i, 4] = 0  # Speed
        c[i, 5] = 0  # Angular speed
    dirExp = c[:, 3]

    for iter in range(iterations):
        # Active or passive influence
        if iter < 0:
            influence_radius = 0
        else:
            influence_radius = i_r

        for i in range(individuals):
            desired_voltage[i, :] = [orientation_voltage + wn, orientation_voltage + wn]

            # Elements detected
            repulsion_walls = np.zeros(individuals)
            repulsion_detected = np.zeros(individuals)
            orientation_detected = np.zeros(individuals)
            attraction_detected = np.zeros(individuals)
            influence_detected = np.zeros(individuals)
            elements_rx = []
            elements_ry = []
            elements_ox = []
            elements_oy = []
            elements_ax = []
            elements_ay = []
            elements_ix = []
            elements_iy = []

            # perception range
            repulsion_range = 3.14159
            orientation_range = 6.28319
            attraction_range = 0.5235988
            influence_range = 3.14159

            # Verify each sensor for repulsion of walls
            for w in range(5):
                if w == 0:
                    dirObs = c[i, 3] - 3.83972
                else:
                    dirObs = dirObs + 1.91986
                if dirObs < 0:
                    dirObs = dirObs + (2 * math.pi)
                elif dirObs > (2 * math.pi):
                    dirObs = dirObs - (2 * math.pi)
                Dir = [math.cos(dirObs), math.sin(dirObs)]
                limitX = c[i, 0] + (Dir[0] * repulsion_radius)
                limitY = c[i, 1] + (Dir[1] * repulsion_radius)

                # Resulting direction due exploration
                if limitX > area_limits or limitX < 0 or limitY > area_limits or limitY < 0:
                    dirExp[i] = dirObs + (3 * math.pi / 4) + (random.uniform(0, 1) * math.pi / 2)
                    repulsion_walls[i] = repulsion_walls[i] + 1
                    if dirExp[i] > (2 * math.pi):
                        dirExp[i] = dirExp[i] - (2 * math.pi)
                    elif dirExp[i] < 0:
                        dirExp[i] = dirExp[i] + (2 * math.pi)

            for j in range(individuals):
                if i != j:  # It must not be the same
                    # Angle of the individual with respect to other members of the swarm
                    neighbors_angle = math.atan2((c[j, 1] - c[i, 1]), (c[j, 0] - c[i, 0]))
                    if neighbors_angle > (2 * math.pi):
                        neighbors_angle = neighbors_angle - (2 * math.pi)
                    elif neighbors_angle < 0:
                        neighbors_angle = neighbors_angle + (2 * math.pi)

                    # Calculation of angles of repulsion and attraction with respect to other individuals
                    beta = neighbors_angle - c[i, 3]
                    if beta < 0:
                        beta = beta + (2 * math.pi)

                    gamma = c[i, 3] - neighbors_angle
                    if gamma < 0:
                        gamma = gamma + (2 * math.pi)

                    if gamma < beta:
                        delta = gamma
                    else:
                        delta = beta

                    # Calculation of the repulsion distance with respect to other individuals
                    if delta < repulsion_range / 2:
                        repulsion_distance = math.sqrt((c[i, 0] - c[j, 0]) ** 2 + (c[i, 1] - c[j, 1]) ** 2)
                    else:
                        repulsion_distance = math.inf

                    # Calculation of the attraction distance with respect to other individuals
                    if delta < attraction_range / 2:
                        attraction_distance = math.sqrt((c[i, 0] - c[j, 0]) ** 2 + (c[i, 1] - c[j, 1]) ** 2)
                    else:
                        attraction_distance = math.inf

                    # Calculation of the orientation distance with respect to other individuals
                    if delta < orientation_range / 2:
                        orientation_distance = math.sqrt((c[i, 0] - c[j, 0]) ** 2 + (c[i, 1] - c[j, 1]) ** 2)
                    else:
                        orientation_distance = math.inf

                    # Number of individuals detected in the radius of repulsion, orientation and attraction
                    if repulsion_distance <= repulsion_radius:
                        elements_rx.append(math.cos(neighbors_angle))
                        elements_ry.append(math.sin(neighbors_angle))
                        repulsion_detected[i] = repulsion_detected[i] + 1

                    if orientation_radius < attraction_distance <= attraction_radius and repulsion_detected[i] == 0 \
                            and repulsion_walls[i] == 0:
                        elements_ax.append(math.cos(neighbors_angle))
                        elements_ay.append(math.sin(neighbors_angle))
                        attraction_detected[i] = attraction_detected[i] + 1

                    if repulsion_radius < orientation_distance <= orientation_radius and \
                            repulsion_detected[i] == 0 and repulsion_walls[i] == 0 and attraction_detected[i] == 0:
                        elements_ox.append(math.cos(c[j, 3]))
                        elements_oy.append(math.sin(c[j, 3]))
                        orientation_detected[i] = orientation_detected[i] + 1

            # Angle of influence
            influence_angle = math.atan2((influence_position[1] - c[i, 1]), (influence_position[0] - c[i, 0]))
            if influence_angle < 0:
                influence_angle = influence_angle + (2 * math.pi)
            elif influence_angle > (2 * math.pi):
                influence_angle = influence_angle - (2 * math.pi)

            # Calculation of influence angles
            i_beta = influence_angle - c[i, 3]
            if i_beta < 0:
                i_beta = i_beta + (2 * math.pi)

            i_gamma = c[i, 3] - influence_angle
            if i_gamma < 0:
                i_gamma = i_gamma + (2 * math.pi)

            if i_gamma < i_beta:
                i_delta = i_gamma
            else:
                i_delta = i_beta

            # Noise for distance value in influence (standard distribution) of 5%
            ds1 = 0
            ds2 = 0
            for n in range(12):
                ds1 = ds1 + random.uniform(-1, 1)
            for m in range(6):
                ds2 = ds2 + random.uniform(-1, 1)
            influence_noise = (ds1 - ds2) * 0.05

            # Calculated influence distance
            if i_delta < influence_range / 2:
                influence_distance = math.sqrt(
                    (influence_position[0] - c[i, 0]) ** 2 + (influence_position[1] - c[i, 1]) ** 2) + influence_noise
            else:
                influence_distance = math.inf

            if influence_distance <= influence_radius and repulsion_detected[i] == 0 and repulsion_walls[i] == 0:
                elements_ix.append(math.cos(influence_angle))
                elements_iy.append(math.sin(influence_angle))
                influence_detected[i] = influence_detected[i] + 1

            # f1 - location time of the swarm
            influence_distance_ = math.sqrt(
                (influence_position[0] - c[i, 0]) ** 2 + (influence_position[1] - c[i, 1]) ** 2)
            if influence_distance_ < influence_radius:
                localized_influence[i, 0] = 1
                failure_radio[iter, i] = 1
            else:
                localized_influence[i, 0] = 0
                failure_radio[iter, i] = 0

            localized_influence_percentage = np.sum(localized_influence) / individuals
            if localized_influence_percentage > 0.6 and localized_influence_flag == 0:
                f1 = iter
                localized_influence_flag += 1

            # Average of detected elements
            if repulsion_detected[i] > 0:
                repulsion_direction = math.atan2((-np.sum(elements_ry)), (-np.sum(elements_rx)))
                if repulsion_direction < 0:
                    repulsion_direction = repulsion_direction + (2 * math.pi)
                elif repulsion_direction > (2 * math.pi):
                    repulsion_direction = repulsion_direction - (2 * math.pi)

            if orientation_detected[i] > 0:
                orientation_direction = math.atan2((np.sum(elements_oy)), (np.sum(elements_ox)))
                if orientation_direction < 0:
                    orientation_direction = orientation_direction + (2 * math.pi)
                elif orientation_direction > (2 * math.pi):
                    orientation_direction = orientation_direction - (2 * math.pi)

            if attraction_detected[i] > 0:
                attraction_direction = math.atan2((np.sum(elements_ay)), (np.sum(elements_ax)))
                if attraction_direction < 0:
                    attraction_direction = attraction_direction + (2 * math.pi)
                elif attraction_direction > (2 * math.pi):
                    attraction_direction = attraction_direction - (2 * math.pi)

            if influence_detected[i] > 0:
                influence_direction = math.atan2((np.sum(elements_iy)), (np.sum(elements_ix)))
                if influence_direction < 0:
                    influence_direction = influence_direction + (2 * math.pi)
                elif influence_direction > (2 * math.pi):
                    influence_direction = influence_direction - (2 * math.pi)

            # Behavior Policies
            # Repulsion rules
            if repulsion_detected[i] > 0:
                state_detected[iter, i] = 1
                desired_voltage[i, :] = [repulsion_voltage + wn, repulsion_voltage + wn]
                xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(repulsion_direction)
                yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(repulsion_direction)
                c[i, 3] = math.atan2(yT, xT)
                dirExp[i] = repulsion_direction

            if repulsion_walls[i] > 0:
                state_detected[iter, i] = 1

                # Influence rules
            if influence_detected[i] > 0 and repulsion_detected[i] == 0 and repulsion_walls[i] == 0:
                i_n = normalize(influence_distance, 0, influence_radius)
                influence_voltage = denormalize(i_n, repulsion_voltage, attraction_voltage)
                if attraction_detected[i] == 0:
                    state_detected[iter, i] = 4
                    desired_voltage[i, :] = [influence_voltage + wn, influence_voltage + wn]
                    xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(influence_direction)
                    yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(influence_direction)
                    c[i, 3] = math.atan2(yT, xT)
                    dirExp[i] = influence_direction
                else:
                    state_detected[iter, i] = 4
                    desired_voltage[i, :] = [influence_voltage + wn, influence_voltage + wn]
                    xT = 0.5 * math.cos(c[i, 3]) + 0.25 * math.cos(influence_direction) + 0.25 * math.cos(
                        attraction_direction)
                    yT = 0.5 * math.sin(c[i, 3]) + 0.25 * math.sin(influence_direction) + 0.25 * math.sin(
                        attraction_direction)
                    c[i, 3] = math.atan2(yT, xT)
                    dirExp[i] = math.atan2((math.sin(influence_direction) + math.sin(attraction_direction)),
                                           (math.cos(influence_direction) + math.cos(attraction_direction)))

            # Attraction rules
            if attraction_detected[i] > 0 and repulsion_detected[i] == 0 and repulsion_walls[i] == 0 and \
                    orientation_detected[i] == 0 and \
                    influence_detected[i] == 0:
                state_detected[iter, i] = 2
                desired_voltage[i, :] = [attraction_voltage + wn, attraction_voltage + wn]
                xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(attraction_direction)
                yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(attraction_direction)
                c[i, 3] = math.atan2(yT, xT)
                dirExp[i] = attraction_direction

            # Orientation rules
            if orientation_detected[i] > 0 and repulsion_detected[i] == 0 and repulsion_walls[i] == 0 and \
                    influence_detected[i] == 0:
                if attraction_detected[i] == 0:
                    state_detected[iter, i] = 3
                    desired_voltage[i, :] = [orientation_voltage + wn, orientation_voltage + wn]
                    xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(orientation_direction)
                    yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(orientation_direction)
                    c[i, 3] = math.atan2(yT, xT)
                    dirExp[i] = orientation_direction
                else:
                    state_detected[iter, i] = 3
                    desired_voltage[i, :] = [orientation_voltage + wn, orientation_voltage + wn]
                    xT = 0.5 * math.cos(c[i, 3]) + 0.25 * math.cos(orientation_direction) + 0.25 * math.cos(
                        attraction_direction)
                    yT = 0.5 * math.sin(c[i, 3]) + 0.25 * math.sin(orientation_direction) + 0.25 * math.sin(
                        attraction_direction)
                    c[i, 3] = math.atan2(yT, xT)
                    dirExp[i] = math.atan2((math.sin(orientation_direction) + math.sin(attraction_direction)),
                                           (math.cos(orientation_direction) + math.cos(attraction_direction)))

            # Out of range
            if attraction_detected[i] == 0 and repulsion_detected[i] == 0 and repulsion_walls[i] == 0 and \
                    orientation_detected[i] == 0 and \
                    influence_detected[i] == 0:
                state_detected[iter, i] = 0
                desired_voltage[i, :] = [orientation_voltage + wn, orientation_voltage + wn]
                xT = 0.5 * math.cos(c[i, 3]) + 0.5 * math.cos(dirExp[i])
                yT = 0.5 * math.sin(c[i, 3]) + 0.5 * math.sin(dirExp[i])
                c[i, 3] = math.atan2(yT, xT)

            report[iter, i, 0] = c[i, 0]
            report[iter, i, 1] = c[i, 1]
            report[iter, i, 2] = c[i, 2]
            report[iter, i, 3] = c[i, 3]
            report[iter, i, 4] = (c[i, 3] * 180) / math.pi
            report[iter, i, 5] = c[i, 4]
            report[iter, i, 6] = c[i, 5]
            report[iter, i, 7] = state_detected[iter, i]

            c_past = c[i, :]
            cs[i, :] = movement(c[i, :], desired_voltage[i, :].reshape(2, 1))
            c[i, :] = cs[i, :]

            if c[i, 0] < (0 + repulsion_radius) or c[i, 0] > (area_limits - repulsion_radius) or c[i, 1] < (
                    0 + repulsion_radius) or c[i, 1] > (area_limits - repulsion_radius):
                c[i, :] = c_past

            # this avoids an infinite increment of radians
            if c[i, 3] > (2 * math.pi):
                c[i, 3] = c[i, 3] - (2 * math.pi)
            elif c[i, 3] < 0:
                c[i, 3] = c[i, 3] + (2 * math.pi)

        cm_[iter, 0] = sum(report[iter, :, 0]) / individuals
        cm_[iter, 1] = sum(report[iter, :, 1]) / individuals
        cm_[iter, 2] = math.sqrt(sum((report[iter, :, 0] - cm_[iter, 0]) ** 2) / individuals)
        cm_[iter, 3] = math.sqrt(sum((report[iter, :, 1] - cm_[iter, 1]) ** 2) / individuals)
        cm_[iter, 4] = math.pi * cm_[iter, 2] * cm_[iter, 3]

        if cm_[iter, 4] > (area_limits * area_limits) * 0.8:
            cm_[iter, 4] = (area_limits * area_limits) * 0.8

        amount_f3[iter] = sum(failure_radio[iter, :])

        # Simulation
        if animation:
            # plt.figure(figsize=(7, 7), dpi=80)
            # ax = plt.gca()
            x = report[iter, :, 0]
            y = report[iter, :, 1]
            vx = np.cos(report[iter, :, 3])
            vy = np.sin(report[iter, :, 3])
            circle = plt.Circle((influence_position[0], influence_position[1]), influence_radius, color='yellow',
                                alpha=0.1)

            plt.cla()
            ax.add_patch(circle)
            for i in range(individuals):
                if state_detected[iter, i] == 0:  # Out of range
                    plt.quiver(x[i], y[i], vx[i], vy[i], color='dimgray')
                elif state_detected[iter, i] == 1:  # Repulsion
                    plt.quiver(x[i], y[i], vx[i], vy[i], color='red')
                elif state_detected[iter, i] == 2:  # Attraction
                    plt.quiver(x[i], y[i], vx[i], vy[i], color='blue')
                elif state_detected[iter, i] == 3:  # Orientation
                    plt.quiver(x[i], y[i], vx[i], vy[i], color='lime')
                elif state_detected[iter, i] == 4:  # Influence
                    plt.quiver(x[i], y[i], vx[i], vy[i], color='yellow')

            ax.set(xlim=(0, area_limits), ylim=(0, area_limits))
            ax.set_aspect('equal')
            plt.pause(0.000001)

        if iter == 0:
            xn[0] = report[0, :, 0]
            yn[0] = report[0, :, 1]
            vxn[0] = np.cos(report[0, :, 3])
            vyn[0] = np.sin(report[0, :, 3])
        elif iter == index_n:
            xn[1] = report[index_n, :, 0]
            yn[1] = report[index_n, :, 1]
            vxn[1] = np.cos(report[index_n, :, 3])
            vyn[1] = np.sin(report[index_n, :, 3])
        elif iter == iterations - 1:
            xn[2] = report[iterations - 1, :, 0]
            yn[2] = report[iterations - 1, :, 1]
            vxn[2] = np.cos(report[iterations - 1, :, 3])
            vyn[2] = np.sin(report[iterations - 1, :, 3])

    # Data
    for n in range(3):
        # plt.figure()
        # ax = plt.gca()
        # ax.set(xlim=(0, area_limits), ylim=(0, area_limits))
        # ax.set_aspect('equal')
        data[n, 0] = sum(xn[n]) / individuals
        data[n, 1] = sum(yn[n]) / individuals
        data[n, 2] = math.sqrt(sum((xn[n] - data[n, 0]) ** 2) / individuals)
        data[n, 3] = math.sqrt(sum((yn[n] - data[n, 1]) ** 2) / individuals)
        data[n, 4] = math.pi * data[n, 2] * data[n, 3]

        # plt.quiver(xn[n], yn[n], vxn[n], vyn[n], color='dimgray')
        # plt.plot(data[n, 0], data[n, 1], 'xb')
        # ellipse = Ellipse(xy=(data[n, 0], data[n, 1]), width=2 * data[n, 2], height=2 * data[n, 3], edgecolor='blue',
        #                    fc='None', lw=1)
        # ax.add_patch(ellipse)
    # plt.show()

    f2 = sum(cm_[:, 4]) / iterations
    f3 = np.amax(amount_f3)
    optimization_functions[0, 0] = f1
    optimization_functions[1, 0] = f2
    optimization_functions[2, 0] = f3
    optimization_functions[3, 0] = f4

    return optimization_functions

"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

def simulation_mean(X, replicas, iterations, i_r):
    # (out, R, table) = simulation_mean(X, replicas, iterations, i_r)

    optimization_functions_report = np.zeros((replicas, 4))
    optimization_functions_mean = np.zeros((replicas, 4))
    R = np.zeros((4))
    animation = False
    Xs = np.zeros(len(X[:]))
    f1 = np.zeros(len(X[:]))
    f2 = np.zeros(len(X[:]))
    f3 = np.zeros(len(X[:]))
    f4 = np.zeros(len(X[:]))
    
    for x in range(len(X[:])):
        Xs = np.zeros(4)
        
        Xs[0] = X[x][0]
        Xs[1] = X[x][1]
        Xs[2] = X[x][2]
        Xs[3] = int(X[x][3])

        for r in range(replicas):

            optimization_functions = aggregation(Xs, iterations, i_r, animation)
            # print("Progress: ", round(percentage[r + 1], 2), "%")

            for f in range(4):
                optimization_functions_report[r, f] = optimization_functions[f, 0]

        for r in range(replicas):
            optimization_functions_mean[r, :] = optimization_functions_report[r, :]
            for f in range(4):
                optimization_functions_mean[r, f] = sum(optimization_functions_mean[0:r + 1, f]) / (r + 1)

        for i in range(4):
            R[i] = optimization_functions_mean[replicas - 1, i]
    
        f1[x] = R[0]
        f2[x] = R[1]
        f3[x] = R[2]
        f4[x] = R[3]  
    
    return np.column_stack([f1, f2, f3, f4])

"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

# Define a problem class that inherits from Problem
class aggregationTask(Problem):

    # Constructor method with the number of variables n_var
    def __init__(self, n_var=4, **kwargs):
        # Call the super class constructor to initialize the problem
        super().__init__(n_var=n_var, n_obj=4, n_ieq_constr=0, vtype=float, **kwargs)

    # Method to evaluate the problem
    def _evaluate(self, x, out, *args, **kwargs):
        # Fill out with the objectives calculated in the `simulation_mean` function
        replicas = 10
        iterations = 300
        i_r = 3
        out["F"]  = simulation_mean(x, replicas, iterations, i_r)

# Create an instance of the foraging problem with 4 variables 
# repulsion_radius, orientation_radius, attraction_radius, individuals
xl = [0,   0.4, 1,   5]
xu = [0.2, 0.6, 1.2, 10]
problem = aggregationTask(n_var=4, xl=xl, xu=xu)

# Get the reference directions for the problem
ref_dirs = get_reference_directions("uniform", 4, n_partitions=12)

# Create an instance of the MOEAD algorithm with the reference directions, 
# 15 neighbors and a 70% probability of neighbor mating
algorithm = MOEAD(ref_dirs, n_neighbors=15, prob_neighbor_mating=0.7,)

# Minimize the problem with the MOEAD algorithm, 200 generations, seed=1 and verbose=False
res = minimize(problem, algorithm, ('n_gen', 20), seed=1, verbose=False)

# Print the final result
print(res.F)