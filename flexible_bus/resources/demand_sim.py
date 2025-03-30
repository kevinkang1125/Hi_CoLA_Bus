from fileinput import filename
from typing_extensions import Self
import numpy as np
import pandas as pd
import os
import math


def ridership_cal(deviate_1,deviate_2,demand_dist,tol):
    ridership = demand_dist[0] + demand_dist[1]*deviate_1 + demand_dist[2]*((1-tol)**(deviate_1)) + demand_dist[3]*deviate_2+ demand_dist[4]*((1-tol)**(deviate_2))
    
    return ridership

def get_demand(time_period):
    demand_dist = []
    if time_period == 1:
        expected_arrivals = [1.5,0.2,1,0.4,0.6]
    elif time_period == 2:
        expected_arrivals = [0.4,0.2,0.4,0.2,0.2]
    else:
        expected_arrivals = [1.6,0.4,1.4,0.4,0.8]
    for i in range(len(expected_arrivals)):
        demand_dist.append(np.random.poisson(expected_arrivals[i]))
    # print(demand_dist)    
    return demand_dist