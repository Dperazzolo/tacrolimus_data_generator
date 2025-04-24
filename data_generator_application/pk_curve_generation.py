#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 11:18:58 2024

@author: diegoperazzolo

- Data generator Informed PK-model tracolimus. 
- Generate signales of Phramacokinetcis of tracolimus.
- Pharmacokinetics = concentration of drug in the blood during time (absorption, availability, etc.)
- model used: one compartment model with first order elemination.  
- reference parameters: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9106813/pdf/nihms-1768954.pdf
- other paper reference: https://ascpt.onlinelibrary.wiley.com/doi/epdf/10.1002/psp4.12966
- reference Pk equation (BASED ON Oral administration - Plasma concentration (single dose)) : https://pharmacy.ufl.edu/files/2013/01/5127-28-equations.pdf

"""
# libraries: 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import math
import random



"""
Create the class person. 
This class generate objects of persons to measure the clearence, volume and relative ke
"""
class person:
    def __init__(self, age, sex, race, weight, SNP, hemoglobine, albumine, last_dose_time, mg_twice_daily_dose, blood_conc): 
        self.age = age
        self.sex = sex
        self.race = race
        self.weight = weight
        self.SNP = SNP
        self.hemoglobine = hemoglobine
        self.albumine = albumine
        self.last_dose_time = last_dose_time
        self.mg_twice_daily_dose = mg_twice_daily_dose
        self.blood_conc = blood_conc
        self.pk_params = self.clearence_volume_ke_computation()
        
        
    def wald_random_sampling(self,teta):
        """
        Compute random sampling based on Wald confidence interval.
        
        Parameters:
            parameters (dict): Dictionary containing mean, se (standard error), and wald keys.
                - mean (float): Mean of the distribution.
                - se (float): Standard error of the distribution.
                - wald (list): Wald confidence interval [lower_bound, upper_bound].
            num_samples (int): Number of samples to be generated.
        
        Returns:
            list: List of random values sampled from the distribution within the Wald confidence interval.
        """
        # Unpack parameters
        mean = teta["mean"]
        se = teta["se"]
        wald_interval = teta["wald"]
        num_sample_from_paper = 363 # --> Value extracted from the source paper: Entire Cohort (N=363)
        
        # Calculate standard deviation from standard error and number of samples obtained from the paper.
        # You can in this way obtain the correct stadard deviation decided from the paper 
        std_dev = se * np.sqrt(num_sample_from_paper)
                
        # Continue sampling until we collect the desired number of samples
        # Sample a random value within the Wald confidence interval
        while True:
            random_value = np.random.normal(mean, std_dev)
            if wald_interval[0] <= random_value <= wald_interval[1]:
                return random_value
    
    
    # def lognormal_random_sampling(self, teta):
    #     # Extract parameters from dictionary
    #     mean = teta["mean"]  # Original scale mean
    #     se = teta["se"]      # Original scale standard error
        
    #     # Convert mean and se to log-space
    #     #mu_log = np.log(mean**2 / np.sqrt(se**2 + mean**2))
    #     #sigma_log = np.sqrt(np.log(1 + (se**2 / mean**2)))
        
    #     # Sample from log-normal distribution in the original space
    #     #sampled_value = np.random.lognormal(mean=mu_log, sigma=sigma_log)
    #     sampled_value = np.random.lognormal(mean=mean, sigma=se)
    #     return sampled_value
    
    
    def clearence_volume_ke_computation(self):
        """
        Calculate clearence (CL), volume (V) and Ke (ke = CL/V) for each subject 
        """
        # CLEARENCE - CL 
        # set statistical model parameters: 
        # teta_1 = {
        #     "mean": 26.2,
        #     "se": 0.7,
        #     "wald": [24.8, 27.6]
        #     } # --> plain measured Clearence without covariates
        # teta_1 = self.wald_random_sampling(teta_1)#-> NORMAL RANDOM SAMPLING 
        #teta_1 = np.random.lognormal(mean=teta_1['mean'], sigma=teta_1['se'])# -> LOGNORMAL SAMPLING 
        teta_1 = 26.2
        print(f"\nteta 1: {teta_1}")
        
        # teta_2 = {
        #     "mean":0.71 ,
        #     "se":0.05 ,
        #     "wald": [0.61, 0.81]
        #     } # --> SNP stat parameter
        # teta_2 = self.wald_random_sampling(teta_2) #-> NORMAL RANDOM SAMPLING 
        #teta_2 = np.random.lognormal(mean=teta_2['mean'], sigma=teta_2['se'])# -> LOGNORMAL SAMPLING 
        
        teta_2 = 0.71
        print(f"\nteta 2: {teta_2} - SNP: {self.SNP}")
        
        # teta_3 = {
        #     "mean": -0.26,
        #     "se": 0.08,
        #     "wald": [-0.42, -0.01]
        #     } # --> age stat parameter
        # teta_3 = self.wald_random_sampling(teta_3)#-> NORMAL RANDOM SAMPLING 
        #teta_3 = np.random.lognormal(mean=teta_3['mean'], sigma=teta_3['se'])# -> LOGNORMAL SAMPLING 
        teta_3 = -0.26
        print(f"\nteta 3: {teta_3} - age: {self.age}" )
        
        # teta_4 = {
        #     "mean": 0.35,
        #     "se": 0.12,
        #     "wald": [0.11, 0.59]
        #     } # --> albumine stat parameter
        # teta_4 = self.wald_random_sampling(teta_4)#-> NORMAL RANDOM SAMPLING 
        #teta_4 = np.random.lognormal(mean=teta_4['mean'], sigma=teta_4['se'])# -> LOGNORMAL SAMPLING 
        teta_4 = 0.35
        print(f"\nteta 4: {teta_4} - albumine: {self.albumine}")
        
        # teta_5 = {
        #     "mean": -0.29,
        #     "se": 0.09,
        #     "wald": [-0.47, -0.11]
        #     }# --> hemoglobine stat parameter
        # teta_5 = self.wald_random_sampling(teta_5)#-> NORMAL RANDOM SAMPLING 
        #teta_5 = np.random.lognormal(mean=teta_5['mean'], sigma=teta_5['se'])# -> LOGNORMAL SAMPLING 
        teta_5 = -0.29
        print(f"\nteta 5: {teta_5} - hemoglobine: {self.hemoglobine}")
        
        
        # calculate the eta CL: 
        eta_CL = np.random.normal(loc=0, scale=0.408)
        exp_value_clearence = math.exp(eta_CL)
       
        # clearence computation
        clearence = teta_1 * (self.SNP**teta_2) * ((self.age/47)**teta_3) * ((self.albumine/4.1)**teta_4) * ((self.hemoglobine/12.5)**teta_5) * (1+exp_value_clearence)
        
        # Random sampling_of the clearence
        #clearence = random.uniform(28.0, 410.0)
        
        print(f"\n computed clearence: {clearence}")
        
        
        # VOLUME - V: 
        # volume_dict = {
        #     "mean": 1437,
        #     "se": 800,
        #     "wald": [2072, 5379]
        #     }
        
        # # Change the values of mean and standard deviation to mu and sigma of lognorm
        # mu_log = np.log(volume_dict['mean']**2 / np.sqrt(volume_dict['se']**2 + volume_dict['mean']**2))
        # sigma_log = np.sqrt(np.log(1 + (volume_dict['se']**2 / volume_dict['mean']**2)))
        
        # #volume = self.wald_random_sampling(volume)#-> NORMAL RANDOM SAMPLING 
        # volume = np.random.lognormal(mean=mu_log, sigma=sigma_log)# -> LOGNORMAL SAMPLING 
        
        # teta 6 is for the volume 
        teta_6 = 3726
        # calculate the eta V: 
        eta_V = np.random.normal(loc=0, scale=0.653) # --> normalsampling
        #eta_V = np.random.lognormal(mean=0, sigma=0.653) # --> lognorm samopling
        exp_value_volume = math.exp(eta_V)
        
        # calculate the volume
        volume = teta_6 * exp_value_volume
        
        print(f"volume (in L): {volume}")
        
        
        # Compute ke of the subject: 
        ke = clearence/volume
        
        # parameter data of the subject: 
        subject_params = {
            "CL": clearence, 
            "V": volume,
            "ke": ke
            }
        
        return subject_params
        






"""
Function to generate a list of person object with all parameters CL, V and Ke and covariates
"""
def generate_person_samples(n_samples): 
    # create a list containing all the person objects. 
    persons_object_lists = []
    
    age_info = {
        "mean": 45.9,
        "std": 12.7,
        "median": 47.0
        }
    
    weight_info = {
        "mean":82.9,
        "std":20.8, 
        "median": 81.2
        }
    
    # hemoglobine    
    hemoglobine_info = {
        "mean":12.5,
        "std":2.1, 
        "median":12.5
        }
    
    # albumine
    albumine_info = {
        "mean": 4.1,
        "std": 0.4,
        "median": 4.1
        }
    
    # mg_twice_daily_dose
    mg_twice_daily_dose_info = {
        "mean": 3.4,
        "std": 1.8,
        "median": 3
        }
    
    # blood_conc
    blood_conc_info = {
        "mean": 7.6,
        "std": 3.1,
        "median": 7.1
        }
    
    while n_samples > 0: 
        
        print("--------------------------------------------------------------------------")
        print(f"\nI still have to generate this number of samples: {n_samples}")
        # age:
        age = int(np.random.normal(age_info["mean"], age_info["std"]))
        # sex:
        sex = np.random.choice(["male", "female"], p=[0.5, 0.5])
        # etnicity:
        race = np.random.choice(["Caucasian-American", "African-American", "Hispanic", "Asian", "Other"], p=[0.2, 0.2, 0.2, 0.2, 0.2])
        # weight: 
        weight = np.random.normal(weight_info["mean"], weight_info["std"])
        # SNP on CYP3A5 gene: 
        SNP = np.random.choice([1,2,3], p=[0.33, 0.34, 0.33])
        # hemoglobine
        hemoglobine = 0
        while hemoglobine == 0:
            hemoglobine = np.random.normal(hemoglobine_info["mean"], hemoglobine_info["std"])
        #hemoglobine = np.random.normal(hemoglobine_info["mean"], hemoglobine_info["std"])
        # albumine
        albumine = 0
        while albumine == 0:
            albumine = np.random.normal(albumine_info["mean"], albumine_info["std"])
        # last-dose
        last_dose_time = 52
        #mg twice daily dose
        mg_twice_daily_dose = np.random.normal(mg_twice_daily_dose_info["mean"], mg_twice_daily_dose_info["std"])
        # blood concentration
        blood_conc = np.random.normal(blood_conc_info["mean"], blood_conc_info["std"])
        
        # -----------------------------------------------------------
        # Create object and append to the list: 
        p = person(age=age, sex=sex, race=race, weight=weight, SNP=SNP, hemoglobine=hemoglobine, albumine=albumine, last_dose_time=last_dose_time, mg_twice_daily_dose=mg_twice_daily_dose, blood_conc=blood_conc)
        persons_object_lists.append(p)
        # -----------------------------------------------------------    
    
        # decrease counter: 
        n_samples = n_samples-1
        print("--------------------------------------------------------------------------\n")
    
    #return the list of object persons
    return persons_object_lists








"""
Class to generate n possible signals. 
Each object iìof the class can generate signals of different kid of patients, ù
depending on the paramters choosen:
    - dose: dose of tracolimus
    - ka: ka of absorpion decided
    - er_random: random of error 
    
Consideration: 
    - Ka is fixed to one of the following values: 4.5, 3.09 or 0.375.
"""
class generate_pk_signals_tracolimus: 
    def __init__(self, dose, er_random, name, td, tlag): 
        self.dose = dose 
        self.ka = np.random.normal(loc=1, scale=0.5) 
        self.er_random = er_random
        self.name = name
        self.td = td
        self.tlag = tlag
        
        
        
    def __str__(self):
        return f"--- Generator of PK-tracolimus signals called: {self.name} --- \nDOSE: {self.dose} \nKa: {self.ka} \nRandom error: {self.er_random} \nAdministration time: {self.td} \nLag time(tlag): {self.tlag} \nMeasured time points: {self.t_list}"        
    
    
    def single_dose_compute_pk_concentration(self, h, ke, volume):
        # formula reference: https://pharmacy.ufl.edu/files/2013/01/5127-28-equations.pdf
        # FORMULA OF PK: BASED ON Oral administration - Plasma concentration (single dose): https://pharmacy.ufl.edu/files/2013/01/5127-28-equations.pdf
        if h <= (self.td + self.tlag):
            print(f"h: {h} is lower or equal than: {self.td + self.tlag}")
            c=0
        else:     
            c = (self.dose/volume) * (self.ka/(self.ka-ke)) * (math.exp(-ke*(h - self.td - self.tlag)) - math.exp(-self.ka*(h - self.td - self.tlag))) # --> with tlag
        #c = (self.dose/volume) * (self.ka/(self.ka-ke)) * (math.exp(-ke*(h - self.td)) - math.exp(-self.ka*(h - self.td)))
        return c 
    
    
    # compute concentration over time with PK-model
    def concetration_over_time_pk_model(self, p:'person', hours_to_sample: 'int'):
        """
        Parameters: 
            - p: is a person bject.
            - hours_to_sample: int parameter that represents the num of hours of determine concentration
        """
        # List of concentration values over time
        concetration_time_series = []
        
        # array of hours: 
        hours = np.arange(hours_to_sample) # It creates from 0 to hours_to_sample-1
        #hours = hours + 1 # I add 1 to all the element of the list (to have hours from 1 to hours_to_sample)
        ke = p.pk_params["ke"]
        volume = p.pk_params["V"]
        
        for h in hours: 
            #Compute concentration
            c = self.single_dose_compute_pk_concentration(h, ke, volume)
            # append to the list of concentration over time
            concetration_time_series.append(c)
            print(f"\nConcentration at the next: {h}-th hour:  {c}")
            print("\n---------------------------------------------------")
        
        
        return list(hours),concetration_time_series
        









if __name__ == "__main__": 
    
    # create the list of persons
    #list_of_persons = generate_person_samples(10000)
    
    # create the list of persons
    list_of_persons = generate_person_samples(2000)
    
    # create the dataset of the covariates for the 
    # Step 1: Extract attributes and create a list of dictionaries
    data_person = []
    for i,person_i in enumerate(list_of_persons):
        person_dict = {
            'age': person_i.age,
            'sex': person_i.sex,
            'race': person_i.race,
            'weight': person_i.weight,
            'SNP': person_i.SNP,
            'hemoglobine': person_i.hemoglobine,
            'albumine': person_i.albumine,
            'last_dose_time': person_i.last_dose_time,
            'mg_twice_daily_dose': person_i.mg_twice_daily_dose,
            'blood_conc': person_i.blood_conc,
            'CL': person_i.pk_params['CL'],
            'V': person_i.pk_params['V'],
            'ke': person_i.pk_params['ke']
        }
        
        print(f"\n Person ID: {i}")
        print(f"\n clearence: {person_i.pk_params['CL']}")
        data_person.append(person_dict)
        
    # create the relative dataframe
    df_persons = pd.DataFrame(data_person)
    # ensure that every numerical feature is a float
    df_persons[['CL', 'V', 'ke']] = df_persons[['CL', 'V', 'ke']].astype(float)
    
    # Create the csv for the covariates of the persons:    
    #df_persons.to_csv('/Volumes/IMAGES_C_D/LSBU_PK_informed_project/Dataset_pk_signals/0_teta_corrected_log_norm_dataset/lognorm_10k_people_covariates.csv', index=False)
    df_persons.to_csv('/Volumes/IMAGES_C_D/LSBU_PK_informed_project/Dataset_pk_signals/0_teta_corrected_log_norm_dataset/test_dataset_lognorm_2k_people_covariates.csv', index=False)
    #df_persons.to_csv('/Volumes/IMAGES_C_D/LSBU_PK_informed_project/Dataset_pk_signals/Log_norm_datasets/dataset_for_Enrico_lognorm_10k_people_covariates.csv', index=False)
                
    
    
    
    
    
    # Generate a Pk signal generator: 
    """
    tlag from the paper: https://ascpt.onlinelibrary.wiley.com/doi/epdf/10.1002/psp4.12966
    has been fixed to 0.346 h^-1 = this means that to find h is equal to 1/0.346 = 2.89
    """
    g1 = generate_pk_signals_tracolimus(dose=300, er_random=0.10, name="G1_300", td=0, tlag=0.346)
    
   
    # create the empty dataframe to create the final csv for the next 120 hour concentration: 
    df_120 = pd.DataFrame(columns=["ID"] + ['t' + str(i) for i in range(0, 121)])
    
    # create the empty dataframe to create the final csv for the next 48 hours concentration: 
    df_48 = pd.DataFrame(columns=["ID"] + ['t' + str(i) for i in range(0, 49)])
    
    # generate concetration of each one of them
    count = 0
    for idx, per in enumerate(list_of_persons):
        # concentration next 120 hours
        hours_120, concentration_120 = g1.concetration_over_time_pk_model(p=per, hours_to_sample=121)
        # concentration next 48 hours 
        hours_48, concentration_48 = g1.concetration_over_time_pk_model(p=per, hours_to_sample=49)
        
        # add to the dataframe: 
        df_120.loc[idx] = [count] + concentration_120
        df_48.loc[idx] = [count] + concentration_48
        
        #increment ID counter: 
        count += 1 
        
        
    # write the final .csv to create the dataset of the signals
    df_120 = df_120.astype(float)
    df_48 = df_48.astype(float)
    #df_120.to_csv('/Volumes/IMAGES_C_D/LSBU_PK_informed_project/Dataset_pk_signals/0_teta_corrected_log_norm_dataset/lognorm_10k_dataset_120h_concentration.csv', index=False)
    #df_48.to_csv('/Volumes/IMAGES_C_D/LSBU_PK_informed_project/Dataset_pk_signals/0_teta_corrected_log_norm_dataset/lognorm_10k_dataset_48h_concentration.csv', index=False)
    
    df_120.to_csv('/Volumes/IMAGES_C_D/LSBU_PK_informed_project/Dataset_pk_signals/0_teta_corrected_log_norm_dataset/test_data_lognorm_2k_dataset_120h_concentration.csv', index=False)
    df_48.to_csv('/Volumes/IMAGES_C_D/LSBU_PK_informed_project/Dataset_pk_signals/0_teta_corrected_log_norm_dataset/test_data_lognorm_2k_dataset_48h_concentration.csv', index=False)
    
    #df_120.to_csv('/Volumes/IMAGES_C_D/LSBU_PK_informed_project/Dataset_pk_signals/Log_norm_datasets/dataset_for_Enrico_10k_dataset_120h_concentration.csv', index=False)
    #df_48.to_csv('/Volumes/IMAGES_C_D/LSBU_PK_informed_project/Dataset_pk_signals/Log_norm_datasets/dataset_for_Enrico_10k_dataset_48h_concentration.csv', index=False)
  
    


    
    
    
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # ----- Example of signal plotting part --------
    
    #"""
    
    # find person with HIGH ABSORPITION RATE:
    high_ke_signals = [person for person in list_of_persons if person.pk_params['ke'] >= 0.03]#[:3] # you can choose how many you want
    # find person with LOW ABSORPITION RATE:
    low_ke_signals = [person for person in list_of_persons if person.pk_params['ke'] < 0.03]#[:3] # you can choose how many you want
    

    # Compute one signal of pk with covariates: 
    hours, pk_time_serie_example_p1 = g1.concetration_over_time_pk_model(p=high_ke_signals[1], hours_to_sample=48)
    hours, pk_time_serie_example_p2 = g1.concetration_over_time_pk_model(p=high_ke_signals[2], hours_to_sample=48)
    hours, pk_time_serie_example_p3 = g1.concetration_over_time_pk_model(p=high_ke_signals[3], hours_to_sample=48)
    hours, pk_time_serie_example_p4 = g1.concetration_over_time_pk_model(p=low_ke_signals[1], hours_to_sample=48)
    hours, pk_time_serie_example_p5 = g1.concetration_over_time_pk_model(p=low_ke_signals[2], hours_to_sample=48)
    hours, pk_time_serie_example_p6 = g1.concetration_over_time_pk_model(p=low_ke_signals[3], hours_to_sample=48)
    
    min_c = min(pk_time_serie_example_p1 + pk_time_serie_example_p2 + pk_time_serie_example_p3 + pk_time_serie_example_p4 + pk_time_serie_example_p5 + pk_time_serie_example_p6)
    max_c = max(pk_time_serie_example_p1 + pk_time_serie_example_p2 + pk_time_serie_example_p3 + pk_time_serie_example_p4 + pk_time_serie_example_p5 + pk_time_serie_example_p6)
    
    
    # ------------------------------------------------------------------------------------
    # g1 - Plot the data
    plt.figure(figsize=(30, 10))
    plt.plot(hours, pk_time_serie_example_p1, color='blue', label='ke_high_serie_example_p1')
    plt.plot(hours, pk_time_serie_example_p2, color='blue', label='ke_high_serie_example_p2')
    plt.plot(hours, pk_time_serie_example_p3, color='blue', label='ke_high_serie_example_p3')
    plt.plot(hours, pk_time_serie_example_p4, color='red', label='ke_low_serie_example_p4')
    plt.plot(hours, pk_time_serie_example_p5, color='red', label='ke_low_serie_example_p5')
    plt.plot(hours, pk_time_serie_example_p6, color='red', label='ke_low_serie_example_p6')
    
    ke_p1 = high_ke_signals[1].pk_params['ke']
    ke_p2 = high_ke_signals[2].pk_params['ke']
    ke_p3 = high_ke_signals[3].pk_params['ke']
    ke_p4 = low_ke_signals[1].pk_params['ke']
    ke_p5 = low_ke_signals[2].pk_params['ke']
    ke_p6 = low_ke_signals[3].pk_params['ke']

    
    
    # Add the value of ke as text on the graph
    plt.text(hours[-2], pk_time_serie_example_p1[-2], f'ke={ke_p1:.2f}', color='blue', fontsize=20)
    plt.text(hours[-2], pk_time_serie_example_p2[-2], f'ke={ke_p2:.2f}', color='blue', fontsize=20)
    plt.text(hours[-2], pk_time_serie_example_p3[-1], f'ke={ke_p3:.2f}', color='blue', fontsize=20)
    plt.text(hours[-2], pk_time_serie_example_p4[-2], f'ke={ke_p4:.2f}', color='red', fontsize=20)
    plt.text(hours[-2], pk_time_serie_example_p5[-2], f'ke={ke_p5:.2f}', color='red', fontsize=20)
    plt.text(hours[-2], pk_time_serie_example_p6[-2], f'ke={ke_p6:.2f}', color='red', fontsize=20)

    
    # Add labels and title
    plt.xlabel('Hours', fontsize=20)
    plt.ylabel('Tacrolimus Concentration (mg/L)', fontsize=20)
    plt.title('PK tacrolimus elimination rate comparison. Difference high and low ke (high: >= 0.03 Low: < 0.03)', fontsize=20)
    # Add legend
    plt.legend(fontsize=20)
    plt.grid()
    # Adjust x-axis ticks
    plt.xticks(range(min(hours), max(hours) + 1), fontsize=20, rotation=90)
    plt.yticks(np.arange(min_c, max_c, 0.02),fontsize=20)
    # Show plot
    plt.show()
    # ------------------------------------------------------------------------------------
    #"""
    
    # analysis on dataset balance
    print("\n Ke < 0.03 = "+ str((len(low_ke_signals)/len(list_of_persons))*100)+"% del dataset")
    print(" Ke >= 0.03 = "+ str((len(high_ke_signals)/len(list_of_persons))*100)+"% del dataset")
    
    
    
    """
    # ------------------------------------------------------------------------------------
    # g2 - Plot the data
    plt.figure(figsize=(30, 10))
    plt.plot(hours_2, pk_time_serie_example_p1_2, color='red', label='pk_time_serie_example')
    # Add labels and title
    plt.xlabel('Hours')
    plt.ylabel('Tacrolimus Concentration')
    plt.title('Time Series Example')
    # Add legend
    plt.legend(fontsize=20)
    # Adjust x-axis ticks
    plt.xticks(range(min(hours_2), max(hours_2) + 1), fontsize=20)
    plt.yticks(fontsize=25)
    # Show plot
    plt.show()
    # ------------------------------------------------------------------------------------
    """
    
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    
    
    