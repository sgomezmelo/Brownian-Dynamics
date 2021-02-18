import numpy as np
from funcs import *
from forces import *
from TOTAL_FORCE_COMP import *

def calc_trajectory(x, y, alpha, run, params):
    # create settings file of tuneable parameters in same folder as data is saved
    f= open(params['MAIN_PATH'] + params['SUB_PATH'] + "Parameters.txt","w+")
    f.write("{:<50s}{:>20s}{:>5E}\r\n".format("Repulsive energy strength", ' = ', params['rep_ep']))
    f.write("{:<50s}{:>20s}{:>5E}\r\n".format("Attractive energy strength", ' = ', params['att_ep']))
    f.write("{:<50s}{:>20s}{:>5E}\r\n".format("Angular cutoff", ' = ', params['ang_c']))
    f.write("{:<50s}{:>20s}{:>5E}\r\n".format("attractive potential cutoff", ' = ', params['rc']))
    f.write("{:<50s}{:>20s}{:>5E}\r\n".format("attractive cutoff", ' = ', params['att_c']))
    f.write("{:<50s}{:>20s}{:>5E}\r\n".format("number of particles", ' = ', params['nparticles']))
    f.write("{:<50s}{:>20s}{:>5E}\r\n".format("number of simulation steps", ' = ', params['simsteps']))
    f.write("{:<50s}{:>20s}{:>5E}\r\n".format("save intervall", ' = ', params['saveint']))
    f.write("{:<50s}{:>20s}{:>5E}\r\n".format("all data saved?", ' = ', params['savealldata']))
    f.write("{:<50s}{:>20s}{:>5E}\r\n".format("tail?", ' = ', params['tail']))
    f.write("{:<50s}{:>20s}{:>5E}\r\n".format("data for movie?", ' = ', params['graphical']))
    f.write("{:<50s}{:>20s}{:>5E}\r\n".format("boxlength", ' = ', params['boxlength']))
    f.write("{:<50s}{:>20s}{:>5E}\r\n".format("monomer concentration", ' = ', params['cmon']))
    f.close()

    PARTICLE_NUMBER = params['nparticles']
    np.savez(params['MAIN_PATH'] + params['SUB_PATH'] + 'params', params=params)
    '''Step 0 Set up data structure for saving purpose'''
    # simulation time in dimensionless units
    time = []
    # particle position and orientations
    # each entry in the array will be another array corresponding to all positions and orientations at a certain time
    xpos = []
    ypos = []
    orient = []
    delta_x_force = []
    delta_y_force = []
    delta_alpha_torque = []
    delta_x_rand = []
    delta_y_rand = []
    delta_alpha_rand = []
    # patch positions
    p1xpos = []
    p1ypos = []
    p2xpos = []
    p2ypos = []
    # particle and patch distances
    # filled with patch distance matrices
    abs_dist = []
    adp11 = []
    adp12 = []
    adp21 = []
    adp22 = []
    # relative patch orientation
    # filled with theta matrices
    tp11 = []
    tp12 = []
    tp21 = []
    tp22 = []
    # total force acting on the particle
    xforce = []
    yforce = []
    tq = []
    # complex size
    number_open = []
    number_closed = []

    s1xpos = []
    s1ypos = []
    s2xpos = []
    s2ypos = []
    s3xpos = []
    s3ypos = []
    s4xpos = []
    s4ypos = []
    s5xpos = []
    s5ypos = []
    sspxpos = []
    sspypos = []

    txori = []
    tyori = []

    for step in range(params['simsteps']):

        if step % 10000 == 0:
            print("STEP: %d/%d"%(step,params['simsteps']))

        '''Step 1 transform coordinates'''
        sinus_alpha = np.sin(alpha)
        cosinus_alpha = np.cos(alpha)
        p1x = patch1_x_lab(x, y, cosinus_alpha, sinus_alpha, params)
        p1y = patch1_y_lab(x, y, cosinus_alpha, sinus_alpha, params)
        p2x = patch2_x_lab(x, y, cosinus_alpha, sinus_alpha, params)
        p2y = patch2_y_lab(x, y, cosinus_alpha, sinus_alpha, params)
        a1x = patch1_alignment_x_lab(cosinus_alpha, sinus_alpha, params)
        a1y = patch1_alignment_y_lab(cosinus_alpha, sinus_alpha, params)
        a2x = patch2_alignment_x_lab(cosinus_alpha, sinus_alpha, params)
        a2y = patch2_alignment_y_lab(cosinus_alpha, sinus_alpha, params)
        tx = tail_x_lab(cosinus_alpha, sinus_alpha, params)
        ty = tail_y_lab(cosinus_alpha, sinus_alpha, params)

        '''Intermediate step for open system - attempt particle creation at a montecarlo step '''

        N_montecarlo = params['Montecarlo step']

        if step%N_montecarlo == 0 :
            #Attempt particle creation

            #coordinates of attempted particle with patches
            L = params['boxlength']
            x_new = np.random.uniform(0,L)
            y_new = np.random.uniform(0,L)
            alpha_new = np.random.uniform(0,2*np.pi)
            c = np.cos(alpha_new)
            s = np.sin(alpha_new)

            p1x_new = patch1_x_lab(x_new, y_new, c, s, params)
            p1y_new = patch1_y_lab(x_new, y_new, c, s, params)
            p2x_new = patch2_x_lab(x_new, y_new, c, s, params)
            p2y_new = patch2_y_lab(x_new, y_new, c, s, params)

            a1x_new = patch1_alignment_x_lab(c, s, params)
            a1y_new = patch1_alignment_y_lab(c, s, params)
            a2x_new = patch2_alignment_x_lab(c, s, params)
            a2y_new = patch2_alignment_y_lab(c, s, params)

            tx_new = tail_x_lab(c, s, params)
            ty_new = tail_y_lab(c, s, params)

            #print('L',L)
            dx = x-x_new
            dx[np.abs(dx)>L] =  dx[np.abs(dx)>L] - L * np.rint(dx[np.abs(dx)>L]/L)
            dy = y-y_new
            dy[np.abs(dy)>L] =  dy[np.abs(dy)>L] - L * np.rint(dy[np.abs(dy)>L]/L)
        
            d = np.sqrt(dx**2+dy**2)

            dx_p1 = compute_dist_patch_AB(p2x,p1x_new,params)
            dy_p1 = compute_dist_patch_AB(p2y,p1y_new,params)
            d_patches_1 = np.sqrt(dx_p1**2 + dy_p1**2)
            theta_1 =  compute_angle_patch_AB(a2x, a2y, a1x_new, a1y_new)

            dx_p2 = compute_dist_patch_AB(p1x,p2x_new,params)
            dy_p2 = compute_dist_patch_AB(p1y,p2y_new,params)
            d_patches_2 = np.sqrt(dx_p2**2 + dy_p2**2)
            theta_2 =  compute_angle_patch_AB(a2x, a2y, a1x_new, a1y_new)

            cutoff_m = params['RBIG']*2.0

            #Apply a cutoff (which makes sense since exp factor will be too small if distances are too short)

            if (d>cutoff_m).all():
                p_create = create_montecarlo(d,d_patches_1,d_patches_2,theta_1,theta_2,params)
                attempt = np.random.uniform()
                # print('prob create',p_create)
                # print('N',PARTICLE_NUMBER)
                # print(step)
                if p_create>attempt:
                    x = np.concatenate((x,np.array([x_new])))
                    y = np.concatenate((y,np.array([y_new])))
                
                    p1x = np.concatenate((p1x,np.array([p1x_new])))
                    p2x = np.concatenate((p2x,np.array([p2x_new])))
                    p1y = np.concatenate((p1y,np.array([p1y_new])))
                    p2y = np.concatenate((p2y,np.array([p2y_new])))

                    a1x = np.concatenate((a1x,np.array([a1x_new])))
                    a2x = np.concatenate((a2x,np.array([a2x_new])))
                    a1y = np.concatenate((a1y,np.array([a1y_new])))
                    a2y = np.concatenate((a2y,np.array([a2y_new])))

                    tx = np.concatenate((tx,np.array([tx_new])))
                    ty = np.concatenate((ty,np.array([ty_new])))

                    alpha = np.concatenate((alpha,np.array([alpha_new])))

                    sinus_alpha = np.concatenate((sinus_alpha,np.array([s])))
                    cosinus_alpha = np.concatenate((cosinus_alpha,np.array([c])))
                
                    PARTICLE_NUMBER += 1

            # Now attempt a particle annihilation
             # i_a = np.random.choice(range(PARTICLE_NUMBER))
             # x_a = x[i_a]
             # y_a = y[i_a]
             # a_a = alpha[i_a]
             
             # p1x_a = p1x[i_a]
             # p2x_a = p2x[i_a]
             # p1y_a = p1x[i_a]
             # p2y_a = p2x[i_a]

             # a1x_a = a1x[i_a]
             # a1y_a = a1y[i_a]
             # a2x_a = a2x[i_a]
             # a2y_a = a2y[i_a]
             

             # dx_d = x_a - np.delete(x,i_a)
             # dy_d = y_a - np.delete(y,i_a)

             # dist_d = np.sqrt(dx_d**2+dy_d**2)

             # dx_p1 = compute_dist_patch_AB(np.delete(p2x,i_a),p1x_a,params)
             # dy_p1 = compute_dist_patch_AB(np.delete(p2y,i_a),p1y_a,params)
             # d_patches_1 = np.sqrt(dx_p1**2+dy_p1**2)
             # theta_1 =  compute_angle_patch_AB(np.delete(a2x,i_a), np.delete(a2y,i_a), a1x_a, a1y_a)

             # dx_p2 = compute_dist_patch_AB(np.delete(p1x,i_a),p2x_a,params)
             # dy_p2 = compute_dist_patch_AB(np.delete(p1y,i_a),p2y_a,params)
             # d_patches_2 = np.sqrt(dx_p2**2+dy_p2**2)
             # theta_2 =  compute_angle_patch_AB(np.delete(a1x,i_a), np.delete(a1y,i_a), a2x_a, a2y_a)

             # p_destroy = destroy_montecarlo(dist_d,d_patches_1,d_patches_2,theta_1,theta_2,params)
             # print('Prob destroy', p_destroy)
             # print('N',PARTICLE_NUMBER)
             # attempt = np.random.uniform()
             # if p_destroy>attempt:
                #  x = np.delete(x,i_a)
                 # y = np.delete(y,i_a)

                #  p1x = np.delete(p1x,i_a)
                 # p1y = np.delete(p1y,i_a)
                 # p2x = np.delete(p2x,i_a)
                 # p2y = np.delete(p2y,i_a)

                 # a1x = np.delete(a1x,i_a)
                 # a1y = np.delete(a1y,i_a)
                 # a2x = np.delete(a2x,i_a)
                 # a2y = np.delete(a2y,i_a)

                 # tx = np.delete(tx,i_a)
                 # ty = np.delete(ty,i_a)

                 # sinus_alpha = np.delete(sinus_alpha,i_a)
                 # cosinus_alpha = np.delete(cosinus_alpha,i_a)

                 # alpha =  np.delete(alpha,i_a)

                 # PARTICLE_NUMBER -= 1

                
        #END OF MONTECARLO ROUTINE 
             
               


        if params['graphical'] is True:
            s1x, s2x, s3x, s4x, s5x = spheres_x_lab(x, y, cosinus_alpha, sinus_alpha, params)
            s1y, s2y, s3y, s4y, s5y = spheres_y_lab(x, y, cosinus_alpha, sinus_alpha, params)
            sspx = ssp_x_lab(x, y, cosinus_alpha, sinus_alpha, params)
            sspy = ssp_y_lab(x, y, cosinus_alpha, sinus_alpha, params)
        '''Step 11 Fill Data Container at certain Intervalls'''
        if step % params['saveint'] == 0:
            # filling data arrays
            time.append(step * params['tstep'])
            # filling position and orientation coordinates
            xpos.append(x)
            ypos.append(y)
            orient.append(alpha)
            # filling patch positions
            p1xpos.append(p1x)
            p1ypos.append(p1y)
            p2xpos.append(p2x)
            p2ypos.append(p2y)

            # orientation tail
            txori.append(tx)
            tyori.append(ty)
            if params['graphical'] is True:
                s1xpos.append(s1x)
                s1ypos.append(s1y)
                s2xpos.append(s2x)
                s2ypos.append(s2y)
                s3xpos.append(s3x)
                s3ypos.append(s3y)
                s4xpos.append(s4x)
                s4ypos.append(s4y)
                s5xpos.append(s5x)
                s5ypos.append(s5y)
                sspxpos.append(sspx)
                sspypos.append(sspy)

        if params['tail'] is True:

            sspx = ssp_x_lab(x, y, cosinus_alpha, sinus_alpha, params)
            sspy = ssp_y_lab(x, y, cosinus_alpha, sinus_alpha, params)

            param_tuple2 = (x, y, alpha, p1x, p1y, p2x, p2y, a1x, a1y, a2x, a2y, sspx, sspy)
            
            (total_force_x,
             total_force_y,
             total_f_att_x_patch1,
             total_f_att_x_patch2,
             total_f_att_y_patch1,
             total_f_att_y_patch2,
             abs_dist_12,
             abs_dist_21,
             theta_12,
             theta_21,
             absolute_dist_matrix,
             total_f_tail_x,
             total_f_tail_y,
             total_orientational_torque) = total_force_comp_with_tail(*param_tuple2, params)

        else:
            param_tuple1 = (x, y, alpha, p1x, p1y, p2x, p2y, a1x, a1y, a2x, a2y)
            (total_force_x,
             total_force_y,
             total_f_att_x_patch1,
             total_f_att_x_patch2,
             total_f_att_y_patch1,
             total_f_att_y_patch2,
             abs_dist_12,
             abs_dist_21,
             theta_12,
             theta_21,
             absolute_dist_matrix,
             total_orientational_torque) = total_force_comp(*param_tuple1, params)


        '''Step 6 Transform forces into PFF'''
        total_force_x_pff = force_x_pff(total_force_x, total_force_y, cosinus_alpha, sinus_alpha)
        total_force_y_pff = force_y_pff(total_force_x, total_force_y, cosinus_alpha, sinus_alpha)

        '''Step 7 Calculate Torque on each Patch (in PFF Coords)'''
        total_torque_patch1 = torque_in_pff(total_f_att_x_patch1, total_f_att_y_patch1,
                                            cosinus_alpha, sinus_alpha,
                                            params['p1x'], params['p1y'])
        total_torque_patch2 = torque_in_pff(total_f_att_x_patch2, total_f_att_y_patch2,
                                            cosinus_alpha, sinus_alpha,
                                            params['p2x'], params['p2y'])

        if params['tail'] is True:
            torque_on_tail = torque_in_pff(total_f_tail_x, total_f_tail_y,
                                           cosinus_alpha, sinus_alpha,
                                           params['sspx'], params['sspy'])
        else:
            torque_on_tail = np.array([0.0]*PARTICLE_NUMBER)

        total_torque = total_torque_patch1 + total_torque_patch2 + torque_on_tail + total_orientational_torque



        '''Collect Cluster Statistics'''
        if step % params['saveint'] == 0:
            number_of_complexes = comp_complexes(abs_dist_12, abs_dist_21, theta_12, theta_21, params)
            open_complexes = number_of_complexes[0]
            closed_complexes = number_of_complexes[1]

        '''Step 8 Generate Random Displacement in Terms of PFF Coords'''
        # g_x = np.array([0] * PARTICLE_NUMBER)
        # g_y = np.array([0] * PARTICLE_NUMBER)
        # g_t = np.array([0] * PARTICLE_NUMBER)
        g_x = np.random.normal(0, params['sdx'], PARTICLE_NUMBER)
        g_y = np.random.normal(0, params['sdy'], PARTICLE_NUMBER)
        g_t = np.random.normal(0, params['sdr'], PARTICLE_NUMBER)


        '''Step 9 Update Positions and Orientations according to Langevin Equation'''
        old_x = x
        dx_force = (cosinus_alpha * params['beta'] * params['Dx'] * params['tstep'] * total_force_x_pff
                    - sinus_alpha * params['beta'] * params['Dy'] * params['tstep'] * total_force_y_pff)
        dx_rand = cosinus_alpha * g_x - sinus_alpha * g_y
        # x = (x + cosinus_alpha * (params['beta'] * params['Dx'] * params['tstep'] * total_force_x_pff + g_x)
        #      - sinus_alpha * (params['beta'] * params['Dy'] * params['tstep'] * total_force_y_pff + g_y))
        old_y = y
        dy_force = (sinus_alpha * params['beta'] * params['Dx'] * params['tstep'] * total_force_x_pff
                    + cosinus_alpha * params['beta'] * params['Dy'] * params['tstep'] * total_force_y_pff)
        dy_rand = sinus_alpha * g_x + cosinus_alpha * g_y
        # y = (y + sinus_alpha * (params['beta'] * params['Dx'] * params['tstep'] * total_force_x_pff + g_x)
        #      + cosinus_alpha * (params['beta'] * params['Dy'] * params['tstep'] * total_force_y_pff + g_y))
        old_alpha = alpha
        alpha_torque = params['beta'] * params['Dr'] * params['tstep'] * total_torque
        alpha_rand = g_t
        # alpha = alpha + params['beta'] * params['Dr'] * params['tstep'] * total_torque + g_t

        if step % params['saveint'] == 0:
            # filling relative patch distances
            abs_dist.append(absolute_dist_matrix)
            # adp11.append(abs_dist_11)
            adp12.append(abs_dist_12)
            adp21.append(abs_dist_21)
            # adp22.append(abs_dist_22)
            # filling relative patch orientation
            # tp11.append(theta_11)
            tp12.append(theta_12)
            tp21.append(theta_21)
            # tp22.append(theta_22)
            # filling forces
            xforce.append(total_force_x)
            yforce.append(total_force_y)
            tq.append(total_torque)
            # filling complex counting
            number_open.append(open_complexes)
            number_closed.append(closed_complexes)

            #displacements in orient and positions
            delta_x_force.append(dx_force)
            delta_y_force.append(dy_force)
            delta_alpha_torque.append(alpha_torque)
            delta_x_rand.append(dx_rand)
            delta_y_rand.append(dy_rand)
            delta_alpha_rand.append(alpha_rand)





        x = old_x + dx_force + dx_rand
        y = old_y + dy_force + dy_rand
        alpha = old_alpha + alpha_torque + alpha_rand
         #print(x)
         #print(y)
         
        '''Step 10 Apply Periodic Boundary Conditions to Particle Coordinates'''
        x[x > params['boxlength']] = x[x > params['boxlength']] - params['boxlength']
        x[x < 0] = x[x < 0] + params['boxlength']
        y[y > params['boxlength']] = y[y > params['boxlength']] - params['boxlength']
        y[y < 0] = y[y < 0] + params['boxlength']



    # preparing data for saving
    t = np.array(time)

    xpos = np.array(xpos)
    ypos = np.array(ypos)
    ori = np.array(orient)

    p1xpos = np.array(p1xpos)
    p1ypos = np.array(p1ypos)
    p2xpos = np.array(p2xpos)
    p2ypos = np.array(p2ypos)

    ad = np.array(abs_dist)
    adp11 = np.array(adp11)
    adp12 = np.array(adp12)
    adp21 = np.array(adp21)
    adp22 = np.array(adp22)

    tp11 = np.array(tp11)
    tp12 = np.array(tp12)
    tp21 = np.array(tp21)
    tp22 = np.array(tp22)

    xf = np.array(xforce)
    yf = np.array(yforce)

    tq = np.array(tq)

    oc = np.array(number_open)
    cc = np.array(number_closed)

    dx_f = np.array(delta_x_force)
    dy_f = np.array(delta_y_force)
    dori_t = np.array(delta_alpha_torque)

    dx_r = np.array(delta_x_rand)
    dy_r = np.array(delta_y_rand)
    dori_r = np.array(delta_alpha_rand)

    s1xpos = np.array(s1xpos)
    s1ypos = np.array(s1ypos)
    s2xpos = np.array(s2xpos)
    s2ypos = np.array(s2ypos)
    s3xpos = np.array(s3xpos)
    s3ypos = np.array(s3ypos)
    s4xpos = np.array(s4xpos)
    s4ypos = np.array(s4ypos)
    s5xpos = np.array(s5xpos)
    s5ypos = np.array(s5ypos)
    sspxpos = np.array(sspxpos)
    sspypos = np.array(sspypos)

    txori = np.array(txori)
    tyori = np.array(tyori)

    print(".....START SAVING.....")
    if params['savealldata'] is True:
        np.savez(params['MAIN_PATH'] + params['SUB_PATH'] + 'time%d'%run, t=t)
        np.savez(params['MAIN_PATH'] + params['SUB_PATH'] + 'part_pos_and_ori%d'%run, xpos=xpos, ypos=ypos, ori=ori)
        np.savez(params['MAIN_PATH'] + params['SUB_PATH'] + 'patch_pos%d'%run, p1xpos=p1xpos, p1ypos=p1ypos, p2xpos=p2xpos, p2ypos=p2ypos)
        np.savez(params['MAIN_PATH'] + params['SUB_PATH'] + 'abs_dist%d'%run, ad=ad, adp11=adp11, adp12=adp12, adp21=adp21, adp22=adp22)
        np.savez(params['MAIN_PATH'] + params['SUB_PATH'] + 'theta_patches%d'%run, tp11=tp11, tp12=tp12, tp21=tp21, tp22=tp22)
        np.savez(params['MAIN_PATH'] + params['SUB_PATH'] + 'total_force%d'%run, xf=xf, yf=yf)
        np.savez(params['MAIN_PATH'] + params['SUB_PATH'] + 'total_torque%d'%run, tq=tq)
        np.savez(params['MAIN_PATH'] + params['SUB_PATH'] + 'open%d'%run, oc=oc)
        np.savez(params['MAIN_PATH'] + params['SUB_PATH'] + 'closed%d'%run, cc=cc)
        np.savez(params['MAIN_PATH'] + params['SUB_PATH'] + 'displ%d'%run,dx_f=dx_f, dx_r=dx_r, dy_f=dy_f, dy_r=dy_r, dori_t=dori_t, dori_r=dori_r)
        np.savez(params['MAIN_PATH'] + params['SUB_PATH'] + 'tail%d'%run, txori=txori, tyori=tyori)

        if params['graphical'] is True:
            np.savez(params['MAIN_PATH'] + params['SUB_PATH'] + 'spheres%d'%run,
                     s1xpos=s1xpos, s1ypos=s1ypos,
                     s2xpos=s2xpos, s2ypos=s2ypos,
                     s3xpos=s3xpos, s3ypos=s3ypos,
                     s4xpos=s4xpos, s4ypos=s4ypos,
                     s5xpos=s5xpos, s5ypos=s5ypos,
                     sspxpos=sspxpos, sspypos=sspypos)

    if params['only_cluster_statistics'] is True:
        np.savez(params['MAIN_PATH'] + params['SUB_PATH'] + 'time%d'%run, t=t)
        np.savez(params['MAIN_PATH'] + params['SUB_PATH'] + 'open%d'%run, oc=oc)
        np.savez(params['MAIN_PATH'] + params['SUB_PATH'] + 'closed%d'%run, cc=cc)

    if params['assembly_details'] is True:
        np.savez(params['MAIN_PATH'] + params['SUB_PATH'] + 'time%d'%run, t=t)
        np.savez(params['MAIN_PATH'] + params['SUB_PATH'] + 'abs_dist%d'%run, ad=ad, adp11=adp11, adp12=adp12, adp21=adp21, adp22=adp22)
        np.savez(params['MAIN_PATH'] + params['SUB_PATH'] + 'theta_patches%d'%run, tp11=tp11, tp12=tp12, tp21=tp21, tp22=tp22)
        np.savez(params['MAIN_PATH'] + params['SUB_PATH'] + 'open%d'%run, oc=oc)
        np.savez(params['MAIN_PATH'] + params['SUB_PATH'] + 'closed%d'%run, cc=cc)
    if params['tail_angle'] is True:
        np.savez(params['MAIN_PATH'] + params['SUB_PATH'] + 'time%d'%run, t=t)
        np.savez(params['MAIN_PATH'] + params['SUB_PATH'] + 'tail%d'%run, txori=txori, tyori=tyori)

    print("RUN = ", run, "......SAVING SUCCESSFUL IN %s"%(params['MAIN_PATH'] + params['SUB_PATH']))

    return oc, cc, t
