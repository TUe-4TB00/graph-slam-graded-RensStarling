import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_landmark_measurement(graph, initial_estimate, result):
    # Determine the correct rotation (bearing) and distance from X(4) to L(2) 
    pose = initial_estimate.atPose2(X(4))
    landmark = initial_estimate.atPoint2(L(2))
    x_X_4 = pose.x()
    y_X_4 = pose.y()
    theta = pose.theta()

    x_L = landmark[0]
    y_L = landmark[1]

    dx = x_L - x_X_4
    dy = y_L - y_X_4

    distance = np.sqrt(dx*dx + dy*dy)
    rotation = np.degrees(np.arctan2(dy, dx) - theta)
    graph.add(gtsam.BearingRangeFactor2D(X(4), L(2), gtsam.Rot2.fromDegrees(rotation), distance, MEASUREMENT_NOISE))
    return graph