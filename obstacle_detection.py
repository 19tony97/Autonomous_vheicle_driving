import numpy as np
from scipy.spatial.distance import euclidean, cdist
from main import BP_LOOKAHEAD_TIME, BP_LOOKAHEAD_BASE, CIRCLE_OFFSETS, CIRCLE_RADII

L = 2

BP_LOOKAHEAD_BASE = int(BP_LOOKAHEAD_BASE)

def next_position(current_x,current_y,yaw,v, delta,L, BP_LOOKAHEAD_TIME, path_iteration):
    x_n = current_x
    y_n = current_y
    yaw_n = yaw 
    delta_t = BP_LOOKAHEAD_TIME

    for i in range(path_iteration+1):
        
        x_n = x_n + v * np.cos(yaw_n)*delta_t
        y_n = y_n + v * np.sin(yaw_n)*delta_t
        yaw_n = yaw_n + ((v * np.tan(delta))/ L) * delta_t

    return x_n, y_n, yaw_n


def circles_for_detection(x,y,yaw,CIRCLE_OFFSETS):
    
    
    current_x, current_y,yaw = x,y,yaw
    
    # get the orientation of the ego-vehicle with formular: position_x_y + distance_between_centroids*cos(yaw)

    x_front = current_x + (CIRCLE_OFFSETS[2]*np.cos(yaw))
    y_front = current_y + (CIRCLE_OFFSETS[2]*np.sin(yaw))

    x_back = current_x + (CIRCLE_OFFSETS[0]*(np.cos(yaw)))
    y_back = current_y + (CIRCLE_OFFSETS[0]*(np.sin(yaw)))
    
  
    center = [0,0,0]
            
    center[0] = [current_x, current_y]
    center[1] = [x_front, y_front]
    center[2] = [x_back, y_back]

    return center

def check_for_obs(obstacles, ego_state, is_collision=False):
    """
    get circles_centers, get obstacle data and check
    whether obstacle location is in distance of radius

    """

    x, y,yaw,v,delta = ego_state[0], ego_state[1], ego_state[2], ego_state[3], ego_state[4]

    for i in range(BP_LOOKAHEAD_BASE):

        if is_collision:            
            break
        
        x_lookahead, y_lookahead, yaw_lookahead = next_position(x,y,yaw,v,delta,L,BP_LOOKAHEAD_TIME, path_iteration = i)

        #centers for ego vehicle
        centers = circles_for_detection(x_lookahead,y_lookahead, yaw_lookahead, CIRCLE_OFFSETS)

        #is_collision = False

        for obstacle in obstacles:
            center_ob = []
            #print(str(obstacle.__class__) == "<class 'carla_server_pb2.Vehicle'>")
            
            if str(obstacle.__class__) == "<class 'carla_server_pb2.Vehicle'>":
                x_ob_veh = obstacle.transform.location.x
                y_ob_veh = obstacle.transform.location.y
                yaw_ob_veh = obstacle.transform.rotation.yaw
                v_ob_veh = obstacle.forward_speed

                # position of obstacle
                xn_ob,yn_ob,yawn_ob = next_position(x_ob_veh,y_ob_veh,yaw_ob_veh,v_ob_veh,delta,L,BP_LOOKAHEAD_TIME, path_iteration=i)
            # circle centers of other vehicles
                center_ob = circles_for_detection(xn_ob, yn_ob, yawn_ob, CIRCLE_OFFSETS)
            else:
                x_ob_ped = obstacle.transform.location.x
                y_ob_ped = obstacle.transform.location.y
                yaw_ob_ped = obstacle.transform.rotation.yaw
                v_ob_ped = obstacle.forward_speed
                
                xn_ob_ped, yn_ob_ped, yawn_ob_ped = next_position(x_ob_ped, y_ob_ped, yaw_ob_ped, v_ob_ped,delta,L,BP_LOOKAHEAD_TIME, path_iteration=i)
                center_ob = [[xn_ob_ped, yn_ob_ped]]
            dists = cdist(centers,center_ob, euclidean)
            
             

            if np.any(dists <= CIRCLE_RADII[0]):  
                is_collision = True
                #print(dists[np.where([dist <= CIRCLE_RADII[0] for dist in dists])] )
                print("detected collision: ", is_collision)

                
                break
            
       

    return is_collision


