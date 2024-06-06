from two_layer_cpg_sns_network import build_net
import mujoco
import numpy as np
import traceback
import mediapy as media
import matplotlib.pyplot as plt
import pandas as pd


def mujoco_model(xml_path):
    """
    loads in the mujoco model to be used as the physics model
    :param xml_path: path to the xml_l mujoco model
    :return:
    """

    # load in the mujoco model and simulation
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    return model, data

def stim_to_act(stim):
    """
    converts from a neural potential to a muscle activation between 0 and 1
    :param stim: MN potential in mV
    :return: act: muscle activation between 0 and 1
    """
    # converted the stim2tenstion curve in animatlab
    steepness = 0.121465
    x_offset = -50
    y_offset = -0.002297
    amp = 1.0
    act = amp/(1 + np.exp(steepness*(x_offset-stim))) + y_offset
    act = np.clip(act, 0,1)
    return act

def run_sims(num_steps, xml_path, L_cpg_inputs, R_cpg_inputs):
    """
    runs the neural and physics models together for a specified number of time
    :param num_steps: number of steps taken in the simulations
    :param modeparams: takes an input of both the synaptic inputs and the gains for Ia feedback
                         g_syns = modeparams[:12]
                         gains = modeparams[12:]*1000

    :param xml_path: path to the Mujoco .xml_l file for the physics simulation
    :return: up to the user
    """

    mujoco_dt = 0.000075
    sns_dt = mujoco_dt *1000

    mujoco_sim, mujoco_data = mujoco_model(xml_path)
    mujoco_sim.opt.timestep = mujoco_dt

    L_sns_model = build_net(dt=sns_dt)
    R_sns_model = build_net(dt=sns_dt)
  
    # initializing vectors
    t = np.arange(0, num_steps)
    time = np.zeros([len(t)])

    num_outputs = L_sns_model.num_outputs
    L_sns_sim_data = np.zeros([len(t), num_outputs])
    L_sns_sim_data[0] = [-100.0, -100.0, -60, -60, -60, -60, -60, -60, -60, -60, -60, -60]

    R_sns_sim_data = np.zeros([len(t), num_outputs])
    R_sns_sim_data[0] = [-100.0, -100.0, -60, -60, -60, -60, -60, -60, -60, -60, -60, -60]

    # get the feedback for the first time_step
    num_inputs = L_sns_model.num_inputs
    L_sns_inputs = np.concatenate([L_cpg_inputs[0,:], np.zeros(num_inputs-2)])
    R_sns_inputs = np.concatenate([R_cpg_inputs[0,:], np.zeros(num_inputs-2)])

    # finding joint indices
    num_joints = mujoco_data.qpos.shape[0]
    if num_joints > 11:
        L_hip_joint_ind = mujoco.mj_name2id(mujoco_sim, mujoco.mjtObj.mjOBJ_JOINT, 'L_hip_flx')+6
        L_knee_joint_ind = mujoco.mj_name2id(mujoco_sim, mujoco.mjtObj.mjOBJ_JOINT, 'L_knee_flx')+6
        L_ankle_joint_ind = mujoco.mj_name2id(mujoco_sim, mujoco.mjtObj.mjOBJ_JOINT, 'L_ankle_flx')+6
        R_hip_joint_ind = mujoco.mj_name2id(mujoco_sim, mujoco.mjtObj.mjOBJ_JOINT, 'R_hip_flx')+6
        R_knee_joint_ind = mujoco.mj_name2id(mujoco_sim, mujoco.mjtObj.mjOBJ_JOINT, 'R_knee_flx')+6
        R_ankle_joint_ind = mujoco.mj_name2id(mujoco_sim, mujoco.mjtObj.mjOBJ_JOINT, 'R_ankle_flx')+6
    else:
        L_hip_joint_ind = mujoco.mj_name2id(mujoco_sim, mujoco.mjtObj.mjOBJ_JOINT, 'L_hip_flx')
        L_knee_joint_ind = mujoco.mj_name2id(mujoco_sim, mujoco.mjtObj.mjOBJ_JOINT, 'L_knee_flx')
        L_ankle_joint_ind = mujoco.mj_name2id(mujoco_sim, mujoco.mjtObj.mjOBJ_JOINT, 'L_ankle_flx')
        R_hip_joint_ind = mujoco.mj_name2id(mujoco_sim, mujoco.mjtObj.mjOBJ_JOINT, 'R_hip_flx')
        R_knee_joint_ind = mujoco.mj_name2id(mujoco_sim, mujoco.mjtObj.mjOBJ_JOINT, 'R_knee_flx')
        R_ankle_joint_ind = mujoco.mj_name2id(mujoco_sim, mujoco.mjtObj.mjOBJ_JOINT, 'R_ankle_flx')

    #finding muscle indices
    L_hip_flx = mujoco.mj_name2id(mujoco_sim, mujoco.mjtObj.mjOBJ_ACTUATOR, 'L_hip_flx')
    L_hip_ext = mujoco.mj_name2id(mujoco_sim, mujoco.mjtObj.mjOBJ_ACTUATOR, 'L_hip_ext')
    L_knee_flx = mujoco.mj_name2id(mujoco_sim, mujoco.mjtObj.mjOBJ_ACTUATOR, 'L_knee_flx')
    L_knee_ext = mujoco.mj_name2id(mujoco_sim, mujoco.mjtObj.mjOBJ_ACTUATOR, 'L_knee_ext')
    L_ankle_flx = mujoco.mj_name2id(mujoco_sim, mujoco.mjtObj.mjOBJ_ACTUATOR, 'L_ankle_flx')
    L_ankle_ext = mujoco.mj_name2id(mujoco_sim, mujoco.mjtObj.mjOBJ_ACTUATOR, 'L_ankle_ext')
    R_hip_flx = mujoco.mj_name2id(mujoco_sim, mujoco.mjtObj.mjOBJ_ACTUATOR, 'R_hip_flx')
    R_hip_ext = mujoco.mj_name2id(mujoco_sim, mujoco.mjtObj.mjOBJ_ACTUATOR, 'R_hip_ext')
    R_knee_flx = mujoco.mj_name2id(mujoco_sim, mujoco.mjtObj.mjOBJ_ACTUATOR, 'R_knee_flx')
    R_knee_ext = mujoco.mj_name2id(mujoco_sim, mujoco.mjtObj.mjOBJ_ACTUATOR, 'R_knee_ext')
    R_ankle_flx = mujoco.mj_name2id(mujoco_sim, mujoco.mjtObj.mjOBJ_ACTUATOR, 'R_ankle_flx')
    R_ankle_ext = mujoco.mj_name2id(mujoco_sim, mujoco.mjtObj.mjOBJ_ACTUATOR, 'R_ankle_ext')

    L_hip_pos = np.zeros([len(t)])
    L_knee_pos = np.zeros([len(t)])
    L_ankle_pos = np.zeros([len(t)])
    R_hip_pos = np.zeros([len(t)])
    R_knee_pos = np.zeros([len(t)])
    R_ankle_pos = np.zeros([len(t)])

    L_hip_pos[0] = mujoco_data.qpos[L_hip_joint_ind]
    L_knee_pos[0] = mujoco_data.qpos[L_knee_joint_ind]
    L_ankle_pos[0] = mujoco_data.qpos[L_ankle_joint_ind]
    R_hip_pos[0] = mujoco_data.qpos[R_hip_joint_ind]
    R_knee_pos[0] = mujoco_data.qpos[R_knee_joint_ind]
    R_ankle_pos[0] = mujoco_data.qpos[R_ankle_joint_ind]

    make_vid = True
    frames = []
    framerate=60 
    renderer = mujoco.Renderer(mujoco_sim, 920,1280) 

    for i in range(1, num_steps):
        try:
            # take one step in neural sim
            L_sns_sim_data[i, :] = L_sns_model(L_sns_inputs)
            R_sns_sim_data[i, :] = R_sns_model(R_sns_inputs)
    
            # find the muscle excitation based on the mn activity from the previous timestep and set muscle activation (0-1)
            mujoco_data.act[L_hip_ext] = stim_to_act(L_sns_sim_data[i - 1, 0])
            mujoco_data.act[L_hip_flx] = stim_to_act(L_sns_sim_data[i - 1, 1])
            mujoco_data.act[L_knee_ext] = stim_to_act(L_sns_sim_data[i - 1, 2])
            mujoco_data.act[L_knee_flx] = stim_to_act(L_sns_sim_data[i - 1, 3])
            mujoco_data.act[L_ankle_ext] = stim_to_act(L_sns_sim_data[i - 1, 4])
            mujoco_data.act[L_ankle_flx] = stim_to_act(L_sns_sim_data[i - 1, 5])    
            mujoco_data.act[R_hip_ext] = stim_to_act(R_sns_sim_data[i - 1, 0])
            mujoco_data.act[R_hip_flx] = stim_to_act(R_sns_sim_data[i - 1, 1])
            mujoco_data.act[R_knee_ext] = stim_to_act(R_sns_sim_data[i - 1, 2])
            mujoco_data.act[R_knee_flx] = stim_to_act(R_sns_sim_data[i - 1, 3])
            mujoco_data.act[R_ankle_ext] = stim_to_act(R_sns_sim_data[i - 1, 4])
            mujoco_data.act[R_ankle_flx] = stim_to_act(R_sns_sim_data[i - 1, 5]) 

            # take one step in the physics sim
            mujoco.mj_step(mujoco_sim, mujoco_data)
            # record
            L_hip_pos[i] = mujoco_data.qpos[L_hip_joint_ind]
            L_knee_pos[i] = mujoco_data.qpos[L_knee_joint_ind]
            L_ankle_pos[i] = mujoco_data.qpos[L_ankle_joint_ind]
            R_hip_pos[i] = mujoco_data.qpos[R_hip_joint_ind]
            R_knee_pos[i] = mujoco_data.qpos[R_knee_joint_ind]
            R_ankle_pos[i] = mujoco_data.qpos[R_ankle_joint_ind]

            time[i] = mujoco_data.time
            # find the feedback from the physics sim
            L_hip_S_ext = 4.3*np.sign(mujoco_data.actuator_velocity[L_hip_ext])*(np.abs(mujoco_data.actuator_velocity[L_hip_ext])**(0.6)) + 82
            L_hip_S_flx = 4.3*np.sign(mujoco_data.actuator_velocity[L_hip_flx])*(np.abs(mujoco_data.actuator_velocity[L_hip_flx])**(0.6)) + 82
            L_knee_S_ext = 4.3*np.sign(mujoco_data.actuator_velocity[L_knee_ext])*(np.abs(mujoco_data.actuator_velocity[L_knee_ext])**(0.6)) + 82
            L_knee_S_flx = 4.3*np.sign(mujoco_data.actuator_velocity[L_knee_flx])*(np.abs(mujoco_data.actuator_velocity[L_knee_flx])**(0.6)) + 82
            L_ankle_S_ext = 4.3*np.sign(mujoco_data.actuator_velocity[L_ankle_ext])*(np.abs(mujoco_data.actuator_velocity[L_ankle_ext])**(0.6)) + 82
            L_ankle_S_flx = 4.3*np.sign(mujoco_data.actuator_velocity[L_ankle_flx])*(np.abs(mujoco_data.actuator_velocity[L_ankle_flx])**(0.6)) + 82

            R_hip_S_ext = 4.3*np.sign(mujoco_data.actuator_velocity[R_hip_ext])*(np.abs(mujoco_data.actuator_velocity[R_hip_ext])**(0.6)) + 82
            R_hip_S_flx = 4.3*np.sign(mujoco_data.actuator_velocity[R_hip_flx])*(np.abs(mujoco_data.actuator_velocity[R_hip_flx])**(0.6)) + 82
            R_knee_S_ext = 4.3*np.sign(mujoco_data.actuator_velocity[R_knee_ext])*(np.abs(mujoco_data.actuator_velocity[R_knee_ext])**(0.6)) + 82
            R_knee_S_flx = 4.3*np.sign(mujoco_data.actuator_velocity[R_knee_flx])*(np.abs(mujoco_data.actuator_velocity[R_knee_flx])**(0.6)) + 82
            R_ankle_S_ext = 4.3*np.sign(mujoco_data.actuator_velocity[R_ankle_ext])*(np.abs(mujoco_data.actuator_velocity[R_ankle_ext])**(0.6)) + 82
            R_ankle_S_flx = 4.3*np.sign(mujoco_data.actuator_velocity[R_ankle_flx])*(np.abs(mujoco_data.actuator_velocity[R_ankle_flx])**(0.6)) + 82

            L_hip_Ia_feedback = [float(L_hip_S_ext)*1.294834-105.557769, float(L_hip_S_flx)*1.783516-145.162975]
            L_hip_Ib_feedback = [-0.290112*mujoco_data.actuator_force[L_hip_ext]+0.022660, -1.338718*mujoco_data.actuator_force[L_hip_flx]+0.089940]
            L_hip_II_feedback = [1091.147828068907*mujoco_data.actuator_length[L_hip_ext]-15.909879574199181, 2729.856579955123*mujoco_data.actuator_length[L_hip_flx]-88.569597408719915]
            L_knee_Ia_feedback = [float(L_knee_S_ext)*1.547175-126.883498, float(L_knee_S_flx)*0.468797-38.305320]
            L_knee_Ib_feedback = [-0.418349*mujoco_data.actuator_force[L_knee_ext]+0.102588, -0.251010*mujoco_data.actuator_force[L_knee_flx]+0.106523]
            L_ankle_Ia_feedback = [float(L_ankle_S_ext)*2.679300-219.277200, float(L_ankle_S_flx)*1.369509-111.88300]
            L_ankle_Ib_feedback = [-0.789629*mujoco_data.actuator_force[L_ankle_ext]+0.000643, -2.288017*mujoco_data.actuator_force[L_ankle_flx]+0.000893]
            L_ankle_II_feedback = [2769.711851*mujoco_data.actuator_length[L_ankle_flx]-82.017204] # only have flx II feedback

            R_hip_Ia_feedback = [float(R_hip_S_ext)*1.294834-105.557769, float(R_hip_S_flx)*1.783516-145.162975]
            R_hip_Ib_feedback = [-0.290112*mujoco_data.actuator_force[R_hip_ext]+0.022660, -1.338718*mujoco_data.actuator_force[R_hip_flx]+0.089940]
            R_hip_II_feedback = [1091.147828068907*mujoco_data.actuator_length[R_hip_ext]-15.909879574199181, 2729.856579955123*mujoco_data.actuator_length[R_hip_flx]-88.569597408719915]
            R_knee_Ia_feedback = [float(R_knee_S_ext)*1.547175-126.883498, float(R_knee_S_flx)*0.468797-38.305320]
            R_knee_Ib_feedback = [-0.418349*mujoco_data.actuator_force[R_knee_ext]+0.102588, -0.251010*mujoco_data.actuator_force[R_knee_flx]+0.106523]
            R_ankle_Ia_feedback = [float(R_ankle_S_ext)*2.679300-219.277200, float(R_ankle_S_flx)*1.369509-111.88300]
            R_ankle_Ib_feedback = [-0.789629*mujoco_data.actuator_force[R_ankle_ext]+0.000643, -2.288017*mujoco_data.actuator_force[R_ankle_flx]+0.000893]
            R_ankle_II_feedback = [2769.711851*mujoco_data.actuator_length[R_ankle_flx]-82.017204] # only have flx II feedback

            L_sns_inputs = np.concatenate([L_cpg_inputs[i,:],L_hip_Ia_feedback,L_hip_Ib_feedback, L_hip_II_feedback, L_knee_Ia_feedback, L_knee_Ib_feedback, L_ankle_Ia_feedback, L_ankle_Ib_feedback, L_ankle_II_feedback])
            R_sns_inputs = np.concatenate([R_cpg_inputs[i,:],R_hip_Ia_feedback,R_hip_Ib_feedback, R_hip_II_feedback, R_knee_Ia_feedback, R_knee_Ib_feedback, R_ankle_Ia_feedback, R_ankle_Ib_feedback, R_ankle_II_feedback])

            if make_vid == True: 
                if len(frames) < mujoco_data.time * framerate:
                    renderer.update_scene(mujoco_data, camera='fixed')
                    pixels = renderer.render().copy()
                    frames.append(pixels) 
    
            if np.abs(L_sns_sim_data[i,1])>100 or np.abs(L_sns_sim_data[i,0])>100:
                return 200.0
        except:
            traceback.print_exc()


    if make_vid == True:
        media.write_video('rat_hopping.mp4', frames, fps=framerate)

    L_sns_sim_data = L_sns_sim_data.T

    return time, L_sns_sim_data, L_hip_pos, L_knee_pos, L_ankle_pos, R_hip_pos, R_knee_pos, R_ankle_pos

def main():

    end_time = 5 # seconds
    num_steps = int(end_time/0.000075)

    xml_path = 'rat_hindlimb_full_ground.xml'
    # xml_path = 'kaiyu_mujoco_on_ground.xml'

    Iapp = np.zeros([num_steps,2])
    Ipert = np.zeros([num_steps,2])
    Ipert[1,0] = 0 # to start cpg's
    L_inputs = Iapp + Ipert

    Iapp = np.zeros([num_steps,2])
    Ipert = np.zeros([num_steps,2])
    Ipert[1,0] = 0 # to start cpg's
    R_inputs = Iapp + Ipert

    time, L_sns_sim_data, L_hip_pos, L_knee_pos, L_ankle_pos, R_hip_pos, R_knee_pos, R_ankle_pos = run_sims(num_steps, xml_path, L_inputs, R_inputs)

    fig, axs = plt.subplots(3, 1, figsize=(15, 10))
    axs[0].plot(time, L_sns_sim_data[:][6], color='red', label='ext')
    axs[0].plot(time, L_sns_sim_data[:][7], color='green', label='flx', linestyle='-.')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Potential (mV)')
    axs[0].set_title('RG Activity')
    axs[0].legend()

    axs[1].plot(time, L_sns_sim_data[:][8], color='red', label='ext')
    axs[1].plot(time, L_sns_sim_data[:][9], color='green', label='flx', linestyle='-.')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Potential (mV)')
    axs[1].set_title('Hip PF Activity')
    axs[1].legend()

    axs[2].plot(time, L_sns_sim_data[:][10], color='red', label='ext')
    axs[2].plot(time, L_sns_sim_data[:][11], color='green', label='flx', linestyle='-.')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Potential (mV)')
    axs[2].set_title('KA PF Activity')
    axs[2].legend()

    plt.tight_layout()
    plt.savefig('SNS_data.jpeg')
    plt.clf()

    fig2, axs2 = plt.subplots(3, 1, figsize=(15, 10))
    axs2[0].plot(time, L_hip_pos, color='blue', label='left')
    axs2[0].plot(time, R_hip_pos, color='red', linestyle='-.',label='right')      
    axs2[0].set_xlabel('Time')
    axs2[0].set_ylabel('Angle (rad)')
    axs2[0].set_title('Hip')
    axs2[0].legend()

    axs2[1].plot(time, L_knee_pos, color='blue', label='left')
    axs2[1].plot(time, R_knee_pos, color='red', linestyle='-.', label='right')
    axs2[1].set_xlabel('Time')
    axs2[1].set_ylabel('Angle (rad)')
    axs2[1].set_title('Knee')
    axs2[1].legend()

    axs2[2].plot(time, L_ankle_pos, color='blue', label='left')
    axs2[2].plot(time, R_ankle_pos, color='red',linestyle='-.', label='right')
    axs2[2].set_xlabel('Time')
    axs2[2].set_ylabel('Angle (rad)')
    axs2[2].set_title('Ankle')
    axs2[2].legend()

    plt.tight_layout()
    plt.savefig('Joint_data.jpeg')
    plt.clf()


    return

if __name__ == '__main__':
    main()
