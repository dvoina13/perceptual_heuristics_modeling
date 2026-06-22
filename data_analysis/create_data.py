import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu
import random
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from sklearn import svm
from scipy.linalg import solve

import torch
import torch.nn as nn
import torch.optim as optim


def get_neuron_rows(n, filename, directory, list_of_stim, list_of_snr, list_of_pos):
    mat_contents = sio.loadmat(directory + filename)

    #print("NEURON N", n)
    total_rows = 0
    list_of_stim_ = [1 if list_of_stim[i] == "circ" else 0 for i in range(len(list_of_stim)) ]

    rows = []
    for trial in range(28):

        #print("trial", trial)
        spikes = mat_contents["Spike_mat"][0][trial]["spikes"]
        errors = mat_contents["Spike_mat"][0][trial]["errors"]
        rt = mat_contents["Spike_mat"][0][trial]["saccade_t"]
        
        n_trials = spikes.shape[0]
        #print("n_trials", n_trials)

        for t in range(n_trials):
            
            if errors[t][0] == 0:
                choice = list_of_stim_[trial]
            else:
                choice = 1 - list_of_stim_[trial]
                
            rows.append({
                "neuron_id": n,
                "spike_ts": np.array(spikes[t,:]).flatten(),
                "firing": np.array(spikes[t,:]).flatten()[500:750].sum(), #/(850-500),
                "stim": list_of_stim[trial],
                "snr": list_of_snr[trial],
                "pos": list_of_pos[trial],
                "error": errors[t][0],
                "choice": choice,
                "reaction_time": rt[t][0],
                "trial": total_rows
            })

            total_rows += 1

    #print("total_rows", total_rows)
    return rows

def make_data_dictionary(directory):

    df_sessions_m1_phase1 = {}

    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    files_split = [f.split('_') for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    new_files = []
    for i, f in enumerate(files_split):
        new_files.append([])
        new_files[i].append(int(f[0][1:]))
        new_files[i].append(int(f[1][1:]))
    new_files = np.array(new_files)

    n_sessions = len(np.unique(np.array(new_files[:, 1])))

    list_of_stim = ["circ", "circ", "rad", "rad"]*7
    list_of_snr = [1.0]*4 + [0.8]*4 + [0.65]*4 + [0.5]*4 + [0.35]*4 + [0.2]*4 + [0.05]*4
    list_of_pos = ["R", "L"] * 14
    dict_of_neurons = {}
    
    for session in range(1, n_sessions+1):
        ind_s = np.where(new_files[:, 1] == session)[0]
        neurons = new_files[ind_s, 0]
        dict_of_neurons[str(session)] = neurons
        
        #print("neuron number", neurons, len(neurons))
        files_s = [files[i] for i in ind_s]
        #print("session", session, neurons, files_s)

        all_rows = []
        for n, filename in zip(neurons, files_s):
            all_rows.extend(get_neuron_rows(n, filename, directory, list_of_stim, list_of_snr, list_of_pos))
            #print("len all rows", len(all_rows))
        
        df_sessions_m1_phase1[str(session)] = pd.DataFrame(all_rows, columns=["neuron_id", "spike_ts", "firing", "stim", "snr", "pos", "error",    
                                                                              "choice", "reaction_time", "trial"])
    return n_sessions, df_sessions_m1_phase1
        
def find_tuned_neurons(df, session, snr_list=[1.0, 0.8, 0.65, 0.5, 0.35, 0.2, 0.05]):
    
    neurons = np.unique(df[str(session)].neuron_id.values)
    neurons_tuned_circ = []
    neurons_tuned_rad = []
    neurons_untuned = []
    
    for n in neurons:

        circular_firings = []
        radial_firings = []
        for SNR in snr_list:
            circular = df[str(session)][(df[str(session)].stim == 'circ') & (df[str(session)].snr == SNR) & (df[str(session)].neuron_id == n)]
            radial = df[str(session)][(df[str(session)].stim == 'rad') & (df[str(session)].snr == SNR) & (df[str(session)].neuron_id == n)]

            #print(len(circular.firing.values), len(radial.firing.values))
            
            circular_firings += list(circular.firing.values)
            radial_firings += list(radial.firing.values)
        
        """
        if p_value<=0.05:
            if np.array(circular_firings).mean() > np.array(radial_firings).mean():
                neurons_tuned_circ.append(n)
            else:       
                neurons_tuned_rad.append(n)
        else:
            neurons_untuned.append(n)
        """
        stat, p_value1 = mannwhitneyu(circular_firings, radial_firings, alternative='greater')
        stat, p_value2 = mannwhitneyu(radial_firings, circular_firings, alternative='greater')
        #stat, p_value = mannwhitneyu(radial_firings, circular_firings, alternative='two-sided')
        #print("p value", p_value, np.array(circular_firings).mean(), np.array(radial_firings).mean())
        print("p_values", p_value1, p_value2)
        
        if p_value1<=0.05:
                neurons_tuned_circ.append(n)
        if p_value2<=0.05:       
                neurons_tuned_rad.append(n)
        if (p_value1 > 0.05) and (p_value2 >0.05):
            neurons_untuned.append(n)
    return neurons_tuned_circ, neurons_tuned_rad, neurons_untuned


#Q1:What is the imbalance? in toy model: 80% versus 20%
def find_tuned_neurons_function(n_sessions, data_dictionary):
    total_neuron_number = 0
    total_number_tuned_circular = 0
    total_number_tuned_radial = 0
    total_number_untuned = 0
    
    number_tuned_circular = []
    number_tuned_radial = []
    number_untuned = []
    
    neuron_id_tuned_circular = []
    neuron_id_tuned_radial = []
    neuron_id_tuned_untuned = []
    
    percentage_of_interest_circ_vs_rad = []
    percentage_of_interest_stim_vs_untuned = []

    for session in range(1,n_sessions+1):
        print("session", session)
        neurons_tuned_circ, neurons_tuned_rad, neurons_untuned = find_tuned_neurons(data_dictionary, session, snr_list=[1.0])

        print(neurons_tuned_circ)
        print(neurons_tuned_rad)
        print(neurons_untuned)
        total_number_tuned_circular += len(neurons_tuned_circ)
        total_number_tuned_radial += len(neurons_tuned_rad)
        total_number_untuned += len(neurons_untuned)

        number_tuned_circular.append(len(neurons_tuned_circ))
        number_tuned_radial.append(len(neurons_tuned_rad))
        number_untuned.append(len(neurons_untuned))
    
        neurons_tuned_circ = np.array(neurons_tuned_circ) + total_neuron_number
        neurons_tuned_rad = np.array(neurons_tuned_rad) + total_neuron_number
        neurons_untuned = np.array(neurons_untuned) + total_neuron_number
    
        neuron_id_tuned_circular += list(neurons_tuned_circ)
        neuron_id_tuned_radial += list(neurons_tuned_rad)
        neuron_id_tuned_untuned += list(neurons_untuned)
        
        total_neuron_number += len(neurons_tuned_circ) + len(neurons_tuned_rad) + len(neurons_untuned)
        
        if len(neurons_tuned_circ) + len(neurons_tuned_rad) != 0:
            percentage_of_interest_circ_vs_rad.append(len(neurons_tuned_circ)/(len(neurons_tuned_circ) + len(neurons_tuned_rad)))
        else:
            percentage_of_interest_circ_vs_rad.append(0)
        percentage_of_interest_stim_vs_untuned.append((len(neurons_tuned_circ) + len(neurons_tuned_rad))/(len(neurons_tuned_circ) + len(neurons_tuned_rad) + len(neurons_untuned)))
    
    print("total neuron number: ", total_neuron_number)
    print("neurons tuned for circular stim: ", neuron_id_tuned_circular, "neurons tuned for radial stim: ", neuron_id_tuned_radial, "untuned neurons: ", neuron_id_tuned_untuned)
    print("number of neurons tuned for circular stim: ", len(neuron_id_tuned_circular), "number of neurons tuned for radial stim: ", len(neuron_id_tuned_radial), "number of untuned neurons: ", len(neuron_id_tuned_untuned))
    
    print("same thing as above (different way): ", total_number_tuned_circular, total_number_tuned_radial, total_number_untuned)
    print("percentage circular", total_number_tuned_circular/(total_number_tuned_circular + total_number_tuned_radial))
    print("percentage tuned:", (total_number_tuned_circular + total_number_tuned_radial)/(total_number_tuned_circular + total_number_tuned_radial + total_number_untuned))

    return np.array(neuron_id_tuned_circular), np.array(neuron_id_tuned_radial), np.array(neuron_id_tuned_untuned)


def create_pd(n_sessions, data_dictionary):
    df_all_sessions = pd.DataFrame([], columns=["neuron_id", "spike_ts", "firing", "stim", "snr", "pos", "error", "choice", "reaction_time", "trial"])
    session_n = pd.DataFrame([], columns=["session"])

    total_neurons = 0
    for session in range(1,n_sessions+1):
        data_dictionary[str(session)].loc[:, 'neuron_id'] = data_dictionary[str(session)].neuron_id.copy() + total_neurons
        total_neurons += len(np.unique(np.array(data_dictionary[str(session)].neuron_id)))
        
        df_all_sessions = pd.concat([df_all_sessions, data_dictionary[str(session)]], axis=0)
        session_n = pd.concat([session_n, pd.DataFrame([session]*len(data_dictionary[str(session)].neuron_id.values), columns=["session"])], axis=0)

    df_all_sessions = pd.concat([df_all_sessions, session_n], axis=1)

    return df_all_sessions

def create_X_and_y(df_all_sessions, snr_choice, choice, session=1, get_data="saved"):

    if get_data == "saved":

        directory = "/home/dvoina/myproj1/"
        print("this is the file", "X_train_pooya_data_phase1_LF_snr" + str(snr_choice) + "_sessALL.npy")
        
        if choice == "ensemble":
            X_train = np.load(directory + "X_train_pooya_data_phase1_LF_snr" + str(snr_choice) + "_sessALL.npy")
            y_train = np.load(directory + "y_train_pooya_data_phase1_LF_snr" + str(snr_choice) + "_sessALL.npy")
            animal_choice_train = np.load(directory + "choice_train_pooya_data_phase1_LF_snr" + str(snr_choice) + "_sessALL.npy")

            X_test = np.load(directory + "X_test_pooya_data_phase1_LF_snr" + str(snr_choice) + "_sessALL.npy")
            y_test = np.load(directory + "y_test_pooya_data_phase1_LF_snr" + str(snr_choice) + "_sessALL.npy")
            animal_choice_test = np.load(directory + "choice_test_pooya_data_phase1_LF_snr" + str(snr_choice) + "_sessALL.npy")

            list_of_neurons = df_all_sessions[(df_all_sessions.stim == 'circ') & (df_all_sessions.snr == snr_choice)].neuron_id.unique()
        else:

            X_train = np.load(directory + "X_train_pooya_data_phase1_LF_snr" + str(snr_choice) + "_sess" + str(session) + ".npy")
            y_train = np.load(directory + "y_train_pooya_data_phase1_LF_snr" + str(snr_choice) + "_sess" + str(session) + ".npy")
            animal_choice_train = np.load(directory + "choice_train_pooya_data_phase1_LF_snr" + str(snr_choice) + "_sess" + str(session) + ".npy")

            X_test = np.load(directory + "X_test_pooya_data_phase1_LF_snr" + str(snr_choice) + "_sess" + str(session) + ".npy")
            y_test = np.load(directory + "y_test_pooya_data_phase1_LF_snr" + str(snr_choice) + "_sess" + str(session) + ".npy")
            animal_choice_test = np.load(directory + "choice_test_pooya_data_phase1_LF_snr" + str(snr_choice) + "_sess" + str(session) + ".npy")

            list_of_neurons = df_all_sessions[(df_all_sessions.stim == 'circ') & (df_all_sessions.snr == snr_choice) & (df_all_sessions.session == session)].neuron_id.unique()

        return X_train, y_train, animal_choice_train, X_test, y_test, animal_choice_test, list_of_neurons, None
        
    if choice == "ensemble":

        if type(snr_choice) == float:
            snr_choice = [snr_choice]
            
        n_neurons = len(df_all_sessions[(df_all_sessions.stim == "circ") & (df_all_sessions.snr == 1.0)].neuron_id.unique())
        list_of_neurons = df_all_sessions.neuron_id.unique()
        n_trials = 1000
 
        list_of_stim = ["circ", "rad"]
        list_of_stim = list_of_stim * (n_trials//len(list_of_stim))

        print("list of neurons:", np.sort(list_of_neurons))

        X_train = np.zeros((n_trials, n_neurons))
        y_train = np.zeros(n_trials)
        animal_choice = np.zeros(n_trials)
        
        selected_trials = []
        snrs_of_choice = [1.0, 0.8, 0.65]
        
        for trial in range(n_trials):
            print("trial", trial)
            for neuron_ind, neuron in enumerate(np.sort(np.array(list_of_neurons))):
            
                trials = df_all_sessions[(df_all_sessions.stim == list_of_stim[trial]) & (df_all_sessions.snr.isin(snr_choice)) & (df_all_sessions.neuron_id == neuron)].trial
            
                rand_trial = np.random.randint(0,len(trials))    
                selected_trial =  rand_trial #trial % len(trials) #
            
                firing = df_all_sessions[(df_all_sessions.stim == list_of_stim[trial]) & (df_all_sessions.snr.isin(snr_choice)) & (df_all_sessions.neuron_id == neuron)].iloc[selected_trial].firing
                X_train[trial, neuron_ind] = firing
                
                choice = df_all_sessions[(df_all_sessions.stim == list_of_stim[trial]) & (df_all_sessions.snr.isin(snr_choice)) & (df_all_sessions.neuron_id == neuron)].iloc[selected_trial].choice
                animal_choice[trial] = choice
            
                if list_of_stim[trial] == "circ":
                    y_train[trial] = 1
                elif list_of_stim[trial] == "rad":
                    y_train[trial] = 0
                else:
                    print("ERROR")
                    break

                snr = df_all_sessions[(df_all_sessions.stim == list_of_stim[trial]) & (df_all_sessions.snr.isin(snr_choice)) & (df_all_sessions.neuron_id == neuron)].iloc[selected_trial].snr
                if snr in snrs_of_choice:
                    selected_trials.append(trial)

        selected_trials = np.array(selected_trials)
        
    elif choice == "only_session":

        if type(snr_choice) == int:
            snr_choice = [snr_choice]
            
        sess = session

        list_of_neurons_s = df_all_sessions[(df_all_sessions.stim == 'circ') & (df_all_sessions.snr.isin(snr_choice)) & (df_all_sessions.session == sess)].neuron_id.unique()
        n_trials_s_circ = len(df_all_sessions[(df_all_sessions.stim == 'circ') & (df_all_sessions.snr.isin(snr_choice)) & (df_all_sessions.session == sess)].trial.unique()) 
        n_trials_s_rad = len(df_all_sessions[(df_all_sessions.stim == 'rad') & (df_all_sessions.snr.isin(snr_choice)) & (df_all_sessions.session == sess)].trial.unique()) 
        n_trials_s = n_trials_s_circ + n_trials_s_rad
    
        n_neurons_s = len(list_of_neurons_s)
        animal_choice = np.zeros(n_trials_s)
        
        print("list of neurons:", list_of_neurons_s)
        list_of_neurons = list_of_neurons_s
        print("number of trials: ", n_trials_s, "number of neurons: ", n_neurons_s)

        X_train = np.zeros((n_trials_s, n_neurons_s))
        y_train = np.zeros(n_trials_s)

        trial = 0
        selected_trials = []
        snrs_of_choice = [1.0, 0.8, 0.65]
        
        for stimulus in ["circ", "rad"]:
        
            print("stimulus", stimulus)
            trials = df_all_sessions[(df_all_sessions.stim == stimulus) & (df_all_sessions.snr.isin(snr_choice))  & (df_all_sessions.session == sess) & (df_all_sessions.neuron_id == list_of_neurons_s[0])].trial.unique()

            min_trials = np.inf
            for nn in range(n_neurons_s):
                trials_ = df_all_sessions[(df_all_sessions.stim == stimulus) & (df_all_sessions.snr.isin(snr_choice))  & (df_all_sessions.session == sess) & (df_all_sessions.neuron_id == list_of_neurons_s[nn])].trial.unique()
                if len(trials_) <= min_trials:
                    min_trials = len(trials_)
            print("len(trials)", min_trials)
            
            
            for t in range(min_trials):
                selected_trial =  trial
                for neuron_ind, neuron in enumerate(np.sort(np.array(list_of_neurons_s))):
    
                    #print("neuron", neuron)
                    firing = df_all_sessions[(df_all_sessions.stim == stimulus) & (df_all_sessions.snr.isin(snr_choice)) & (df_all_sessions.session == sess) & (df_all_sessions.neuron_id == neuron)].iloc[t].firing
                    X_train[trial, neuron_ind] = firing
    
                    animal_choice[trial] = df_all_sessions[(df_all_sessions.stim == stimulus) & (df_all_sessions.snr.isin(snr_choice)) & (df_all_sessions.session == sess) & (df_all_sessions.neuron_id == neuron)].iloc[t].choice
                    
                    if stimulus == "circ":
                        y_train[trial] = 1
                    elif stimulus == "rad":
                        y_train[trial] = 0
                    else:
                        print("ERROR")
                        break

                trial += 1

                snr = df_all_sessions[(df_all_sessions.stim == stimulus) & (df_all_sessions.snr.isin(snr_choice)) &  (df_all_sessions.session == sess) & (df_all_sessions.neuron_id == neuron)].iloc[t].snr
                if snr in snrs_of_choice:
                    selected_trials.append(trial)

    permuted_trials = np.random.permutation(X_train.shape[0])
    X_train = X_train[permuted_trials,:]
    y_train = y_train[permuted_trials]
    animal_choice = animal_choice[permuted_trials]
    
    X_total = X_train.copy()
    y_total = y_train.copy()
    animal_choice_total = animal_choice.copy()
    
    n = X_train.shape[0]
    X_test = X_train[int(0.8*n):,:]
    y_test = y_train[int(0.8*n):]
    X_train = X_train[:int(0.8*n),:]
    y_train = y_train[:int(0.8*n)]
    animal_choice_train = animal_choice[:int(0.8*n)]
    animal_choice_test = animal_choice[int(0.8*n):]
    
    return X_train, y_train, animal_choice_train, X_test, y_test, animal_choice_test, X_total, y_total, animal_choice_total, list_of_neurons, selected_trials

def recompute_tunings(list_of_neurons_s, tunings):

    neuron_id_tuned_circular, neuron_id_tuned_radial, neuron_id_tuned_untuned = tunings
    
    neurons_circ_new = []
    neurons_rad_new = []
    neurons_untuned_new = []
    for neuron_ind, neuron in enumerate(np.sort(list_of_neurons_s)):
        if neuron in neuron_id_tuned_circular:
            neurons_circ_new.append(neuron_ind)
        elif neuron in neuron_id_tuned_radial:
            neurons_rad_new.append(neuron_ind)
        elif neuron in neuron_id_tuned_untuned:
            neurons_untuned_new.append(neuron_ind)

    return neurons_circ_new, neurons_rad_new, neurons_untuned_new

def run(directory, snr_choice = 1.0, choice = "only_session", session=None, get_data="saved"): #"ensamble"
    
    directory_ = "/home/dvoina/myproj1/pooya_data/" + directory
    n_sessions, data_dictionary = make_data_dictionary(directory_)

    print("n_sessions", n_sessions)
    neuron_id_tuned_circular, neuron_id_tuned_radial, neurons_untuned = find_tuned_neurons_function(n_sessions, data_dictionary)
    tunings = [neuron_id_tuned_circular, neuron_id_tuned_radial, neurons_untuned]
    print("len of diff tuned neurons", len(neuron_id_tuned_circular), len(neuron_id_tuned_radial), len(neurons_untuned))

    print("neuron_id_tuned_circular", neuron_id_tuned_circular)
    df_all_sessions = create_pd(n_sessions, data_dictionary)

    X_train, y_train, animal_choice_train, X_test, y_test, animal_choice_test, X_total, y_total, animal_choice_total, list_of_neurons_s, selected_trials = create_X_and_y(df_all_sessions, snr_choice, choice=choice, session=session, get_data=get_data)
    
    tunings_session = None
    if choice == "only_session":
        neurons_circ_new, neurons_rad_new, neurons_untuned_new = recompute_tunings(list_of_neurons_s, tunings)
        tunings_session = [neurons_circ_new, neurons_rad_new, neurons_untuned_new]
        
    print("shapes of X_train, y_train", X_train.shape, y_train.shape)
    return n_sessions, data_dictionary, df_all_sessions, tunings, tunings_session, X_train, y_train, animal_choice_train, X_test, y_test, animal_choice_test, X_total, y_total, animal_choice_total, selected_trials

#if __name__ == "__main__":
#n_sessions, data_dictionary, df_all_sessions, tunings, tunings_session, X_train, y_train, animal_choice = run()
