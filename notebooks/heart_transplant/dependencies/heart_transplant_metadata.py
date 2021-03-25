heart_transplant_metadata = {
    'thoracic_dgn': {
        'name_long': 'Thoracic diagnosis',
        'type': 'recipient'
    },
    'gender': {
        'name_long': 'Gender',
        'type': 'recipient'
    },
    'abo': {
        'name_long': 'Blood group',
        'type': 'recipient'
    },
    'wgt_kg_tcr': {
        'name_long': "Recipient weight",
        'remove': True,
    },
    'hgt_cm_calc': {
        'name_long': 'Recipient height',
        'type': 'recipient'
    },
    'wgt_kg_calc': {
        'name_long': 'Recipient weight',
        'type': 'recipient'
    },
    'bmi_calc': {
        'name_long': 'BMI recipient',
        'type': 'recipient'
    },
    'wgt_kg_don_calc': {
        'name_long': 'Donor weight',
        'type': 'donor'
    },
    'hgt_cm_don_calc': {
        'name_long': 'Donor height',
        'type': 'donor'
    },
    'organ': {
        'name_long': 'Organ Transplanted',
        'remove': True,
    },
    'ebv_igg_cad_don': {
        'name_long': 'EBV IgG in donor',
        'type': 'donor',
        'na_values': ['U'],
    },
    'iabp_tcr': {
        'name_long': 'Intra-aortic balloon pump',
        'type': 'recipient'
    },
    'inotropes_tcr': {
        'name_long': 'Inotropic Support',
        'type': 'recipient'
    },
    'func_stat_tcr': {
        'name_long': 'Functional status at listing recipient',
        'type': 'recipient',
        'na_values': [998.],
    },
    'diab': {
        'name_long': 'Diabetes',
        'type': 'recipient',
        'na_values': [998.],
    },
    'cereb_vasc': {
        'name_long': 'Cerebrovascular disease',
        'type': 'recipient',
        'na_values': ['U'],
    },
    'most_rcnt_creat': {
        'name_long': 'Creatinine at listing',
        'type': 'recipient'
    },
    'tot_serum_album': {
        'name_long': 'Serum albumin',
        'type': 'recipient'
    },
    'impl_defibril': {
        'name_long': 'Defibrillator in recipient',
        'type': 'recipient',
        'na_values': ['U'],
    },
    'hemo_sys_tcr': {
        'name_long': 'Systolic pulmonary pressure recipient (mmHg)',
        'type': 'recipient'
    },
    'hemo_pa_dia_tcr': {
        'name_long': 'Diastolic pulmonary pressure recipient (mmHg)',
        'type': 'recipient'
    },
    'hemo_pa_mn_tcr': {
        'name_long': 'Mean pulmonary pressure recipient (mmHg)',
        'type': 'recipient'
    },
    'hemo_pcw_tcr': {
        'name_long': 'Mean pulmonary wedge pressure recipient (mmHg)',
        'type': 'recipient'
    },
    'hemo_co_tcr': {
        'name_long': 'Cardiac output recipient (L/min)',
        'type': 'recipient'
    },
    'cig_use': {
        'name_long': 'Cigarette use recipient',
        'type': 'recipient'
    },
    'prior_card_surg_tcr': {
        'name_long': 'Prior cardiac surgery recipient',
        'type': 'recipient',
        'na_values': ['U'],
    },
    'prior_card_surg_type_tcr': {
        'name_long': 'Prior cardiac surgery recipient type',
        'type': 'recipient'
    },
    'ethcat': {
        'name_long': 'Ethnicity',
        'type': 'recipient',
        'na_values': [998],
        'value_map': {
            1: 'white',
            2: 'black',
            4: 'hispanic',
            5: 'asian',
            6: 'native alaska',
            7: 'native hawaii/pacific islander',
            9: 'multiracial',
            998: 'unknown',
        }
    },
    'ventilator_tcr': {
        'name_long': 'Ventilator Support at Listing',
        'type': 'recipient',
        'na_values': ['U'],
    },
    'lvad ever': {
        'name_long': 'LV assist device',
        'type': 'recipient',
        # Missing in some folds
        'remove': True,
    },
    'rvad ever': {
        'name_long': 'RV assist device',
        'type': 'recipient',
        # Missing in some folds
        'remove': True,
    },
    'tah ever': {
        'name_long': 'Total artificial heart',
        'type': 'recipient',
    },
    'med_cond_trr': {
        'name_long': 'Medical condition at transplant',
        'type': 'recipient',
        'value_map': {
            1: 'ICU',
            2: 'hospital not ICU',
            3: 'not hospitalized'
        }
    },
    'ecmo_trr': {
        'name_long': 'ECMO at Transplant',
        'type': 'recipient'
    },
    'creat_trr': {
        'name_long': 'Creatinine at transplant',
        'type': 'recipient'
    },
    'dial_after_list': {
        'name_long': 'Dialysis after listing',
        'type': 'recipient',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'tbili': {
        'name_long': 'Bilirubin—Recipient',
        'type': 'recipient',
        'na_values': ['N', 'U', 'Y'],
    },
    'tbili_don': {
        'name_long': 'Bilirubin—Donor',
        'type': 'recipient',
        'na_values': ['N', 'U', 'Y'],
    },
    'transfusions': {
        'name_long': 'Transfusions since listing',
        'type': 'recipient',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'hbv_core': {
        'name_long': 'HBV core',
        'type': 'recipient',
        'na_values': ['U'],
        'value_map': {
            'N': 'negative',
            'ND': 'not done',
            'P': 'positive',
            'U': 'unknown'
        }
    },
    'hbv_sur_antigen': {
        'name_long': 'HBV surface antigen',
        'type': 'recipient',
        'na_values': ['U'],
        'value_map': {
            'N': 'negative',
            'ND': 'not done',
            'P': 'positive',
            'U': 'unknown'
        }
    },
    'cmv_status': {
        'name_long': 'CMV status',
        'type': 'recipient',
        'na_values': ['U'],
        'value_map': {
            'N': 'negative',
            'ND': 'not done',
            'P': 'positive',
            'U': 'unknown'
        }
    },
    'cmv_igg': {
        'name_long': 'CMV IgG',
        'type': 'recipient',
        'na_values': ['U', 'PD', 'C'],
        'value_map': {
            'N': 'negative',
            'ND': 'not done',
            'P': 'positive',
            'U': 'unknown'
        }
    },
    'cmv_igm': {
        'name_long': 'CMV IgM',
        'type': 'recipient',
        'na_values': ['U'],
        'value_map': {
            'N': 'negative',
            'ND': 'not done',
            'P': 'positive',
            'U': 'unknown'
        }
    },
    'hiv_serostatus': {
        'name_long': 'HIV serostatus',
        'type': 'recipient',
        'na_values': ['U'],
        'value_map': {
            'N': 'negative',
            'ND': 'not done',
            'P': 'positive',
            'U': 'unknown'
        }
    },
    'hcv_serostatus': {
        'name_long': 'HCV serostatus',
        'type': 'recipient',
        'na_values': ['U'],
        'value_map': {
            'N': 'negative',
            'ND': 'not done',
            'P': 'positive',
            'U': 'unknown'
        }
    },
    'ebv_serostatus': {
        'name_long': 'EBV serostatus',
        'type': 'recipient',
        'na_values': ['U'],
        'value_map': {
            'N': 'negative',
            'ND': 'not done',
            'P': 'positive',
            'U': 'unknown'
        }
    },
    'gstatus': {
        'name_long': 'Survival of graft',
        'type': 'matching',
    },
    'gtime': {
        'name_long': 'Time of graft survival',
        'type': 'matching'
    },
    'pstatus': {
        'name_long': 'Patient survival',
        'type': 'matching'
    },
    'ptime': {
        'name_long': 'Time of patient survival',
        'type': 'matching'
    },
    'px_stat': {
        'name_long': 'Patient status',
        'type': 'matching'
    },
    'prev_tx': {
        'name_long': 'Previous transplantation',
        'type': 'recipient'
    },
    'prev_tx_any': {
        'name_long': 'Previous transplantation',
        'type': 'recipient'
    },
    'hist_cocaine_don': {
        'name_long': 'History of cocaine use—Donor',
        'type': 'donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'age_don': {
        'name_long': 'Age donor',
        'type': 'donor'
    },
    'ethcat_don': {
        'name_long': 'Ethnicity donor',
        'type': 'donor',
        'na_values': [998],
        'value_map': {
            1: 'white',
            2: 'black',
            4: 'hispanic',
            5: 'asian',
            6: 'native alaska',
            7: 'native hawaii/pacific islander',
            9: 'multiracial',
            998: 'unknown',
        }
    },
    'hbv_core_don': {
        'name_long': 'HBV core donor',
        'type': 'donor',
        'na_values': ['U', 'PD', 'C'],
        'value_map': {
            'N': 'negative',
            'ND': 'not done',
            'P': 'positive',
            'U': 'unknown',
            'C/I': 'unclear'
        }
    },
    'hbv_sur_antigen_don': {
        'name_long': 'HBV surface antigen donor',
        'type': 'donor',
        'na_values': ['U', 'C'],
        'value_map': {
            'N': 'negative',
            'ND': 'not done',
            'P': 'positive',
            'U': 'unknown',
            'C/I': 'unclear'
        }
    },
    'abo_don': {
        'name_long': 'Blood group donor',
        'type': 'donor'
    },
    'alcohol_heavy_don': {
        'name_long': 'Heavy alcohol use donor',
        'type': 'donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'gender_don': {
        'name_long': 'Gender donor',
        'type': 'donor'
    },
    'hep_c_anti_don': {
        'name_long': 'HCV serostatus donor',
        'type': 'donor',
        'value_map': {
            'N': 'negative',
            'ND': 'not done',
            'P': 'positive',
            'U': 'unknown',
            'I': 'not clear'
        },
        'na_values': ['U', 'PD'],
    },
    'non_hrt_don': {
        'name_long': 'Non-beating heart donor',
        'type': 'donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'negative',
            'ND': 'not done',
            'P': 'positive',
            'U': 'unknown',
            'I': 'not clear'
        },
        # Missing in some folds
        'remove': True,
    },
    'blood_inf_don': {
        'name_long': 'Infection in blood Donor',
        'type': 'donor'
    },
    'bun_don': {
        'name_long': 'Blood urea nitrogen donor',
        'type': 'donor'
    },
    'creat_don': {
        'name_long': 'Creatinine donor',
        'type': 'donor'
    },
    'other_inf_don': {
        'name_long': 'Other infection in donor',
        'type': 'donor'
    },
    'pulm_inf_don': {
        'name_long': 'Pulmonary infection donor',
        'type': 'donor'
    },
    'sgot_don': {
        'name_long': 'AST donor',
        'type': 'donor'
    },
    'sgpt_don': {
        'name_long': 'ALT donor',
        'type': 'donor'
    },
    'urine_inf_don': {
        'name_long': 'Urine infection donor',
        'type': 'donor'
    },
    'vasodil_don': {
        'name_long': 'Vasodilatators use donor',
        'type': 'donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'vdrl_don': {
        'name_long': 'RPR-VDRL serology donor',
        'type': 'donor',
        'na_values': ['U', 'PD', 'C'],
        'value_map': {
            'subdivision N': 'negative',
            'ND': 'not done',
            'P': 'positive',
            'U': 'unknown'
        }
    },
    'clin_infect_don': {
        'name_long': 'Clinical infection donor',
        'type': 'donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'contin_cocaine_don': {
        'name_long': 'Cocaine use in last 6 months donor',
        'type': 'donor',
        'na_values': ['U'],
        'remove': True,
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'contin_oth_drug_don': {
        'name_long': 'Other drug use in last 6 months—Donor',
        'type': 'donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'extracranial_cancer_don': {
        'name_long': 'Extracranial cancer in donor',
        'type': 'donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'hist_alcohol_old_don': {
        'name_long': 'History of alcohol in donor',
        'type': 'donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'cancer_site_don': {
        'name_long': 'Site of cancer donor',
        'type': 'donor',
        'value_map': {
            '1': 'no cancer'
        },
        'na_values': [998., 999.]
    },
    'hist_cig_don': {
        'name_long': 'History of cigarettes use donor',
        'type': 'donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes'
        }
    },
    'hist_hypertens_don': {
        'name_long': 'History of hypertension donor',
        'type': 'donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'hist_iv_drug_old_don': {
        'name_long': 'History of IV drug use donor',
        'type': 'donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'intracranial_cancer_don': {
        'name_long': 'Intracranial cancer donor',
        'type': 'donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'hist_cancer_don': {
        'name_long': 'History of cancer donor',
        'type': 'donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'donor insulin': {
        'name_long': 'Insulin dependence donor',
        'type': 'donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'diabetes_don': {
        'name_long': 'Diabetes donor',
        'type': 'donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'hist_oth_drug_don': {
        'name_long': 'History of other drug abuse donor',
        'type': 'donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'skin_cancer_don': {
        'name_long': 'Skin cancer in donor',
        'type': 'donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'cmv_don': {
        'name_long': 'CMV infection donor',
        'type': 'donor',
        'na_values': ['U', 'C'],
        'value_map': {
            'N': 'negative',
            'ND': 'not done',
            'P': 'positive',
            'U': 'unknown'
        }
    },
    'cod_cad_don': {
        'name_long': 'Cause of death donor',
        'type': 'donor',
        'na_values': [999.0],
        'value_map': {
            1: 'Anoxia',
            2: 'Cerebrovascular/Stroke',
            3: 'Head Trauma',
            4: 'Other',
        }
    },
    'death_mech_don': {
        'name_long': 'Cause of mechanical death donor',
        'type': 'donor',
        'na_values': [997., 995.]
    },
    'bmi_don_calc': {
        'name_long': 'Calculated BMI donor',
        'type': 'donor',
    },
    'multiorg': {
        'name_long': 'Multiorgan transplant',
        'type': 'recipient'
    },
    'abo_mat': {
        'name_long': 'Matched bloodgroup',
        'type': 'matching',
        'na_values': [3],
    },
    'age': {
        'name_long': 'Age—Recipient',
        'type': 'recipient'
    },
    'diag': {
        'name_long': 'Thoracic diagnosis recipient',
        'type': 'recipient'
    },
    'dial_prior_tx': {
        'name_long': 'Dialysis prior to transplantation',
        'type': 'recipient',
        'na_values': ['U'],
    },
    'ischtime': {
        'name_long': 'Ischemic time',
        'type': 'matching'
    },
    'life_sup_trr': {
        'name_long': 'Patient on life support—Recipient',
        'type': 'recipient',
        'value_map': {
            'Y': 'yes',
            'N': 'no'
        },
    },
    'prior_card_surg_trr': {
        'name_long': 'Cardiac surgery between listing and transplant',
        'type': 'recipient',
        'na_values': ['U'],
        'value_map': {
            'Y': 'yes',
            'N': 'no'
        },
    },
    'malig': {
        'name_long': 'Any previous cancer',
        'type': 'recipient',
        'na_values': ['U'],
        'value_map': {
            'Y': 'yes',
            'N': 'no',
            'U': 'unknown'
        },
    },
    'distance': {
        'name_long': 'Distance between donor and recipient center',
        'type': 'matching',
    },
    'vent_support_after_list': {
        'name_long': 'Ventilation support after listing',
        'type': 'recipient',
        'na_values': ['U'],
        'value_map': {
            'Y': 'yes',
            'N': 'no',
            'U': 'unknown'
        },
    },
    'protein_urine': {
        'name_long': 'Proteinuria',
        'type': 'recipient',
        'na_values': ['U'],
        'value_map': {
            'Y': 'yes',
            'N': 'no',
            'U': 'unknown'
        },
    },
    'hist_mi': {
        'name_long': 'History of myocardial infarct',
        'type': 'recipient',
        'na_values': ['U'],
        'value_map': {
            'Y': 'yes',
            'N': 'no',
            'U': 'unknown'
        },
    },
    'lv_eject_meth': {
        'name_long': 'Method measuring LVEF donor',
        'type': 'donor'
    },
    'lv_eject': {
        'name_long': 'LVEF donor',
        'type': 'donor'
    },
    'coronary_angio': {
        'name_long': 'Angiogram in donor',
        'type': 'donor',
        'value_map': {
            1: 'no angio',
            2: 'normal angio',
            3: 'abnormal angio'
        },
    },
    'vessels_50sten': {
        'name_long': 'Number of vessels with 50% stenosis in donor',
        'type': 'donor'
    },
    'biopsy_dgn': {
        'name_long': 'Heart biopsy performed donor',
        'type': 'donor'
    },
    'tattoos': {
        'name_long': 'Tattoos donor',
        'type': 'donor',
        'na_values': ['U'],
        'value_map': {
            'Y': 'yes',
            'N': 'no',
            'U': 'unknown'
        },
    },
    'cdc_risk_hiv_don': {
        'name_long': 'CDC high-risk donor',
        'type': 'donor',
        'na_values': ['U', 'PD', 'C'],
        'value_map': {
            'N': 'negative',
            'ND': 'not done',
            'P': 'positive',
            'U': 'unknown'
        },
        # Missing in some folds
        'remove': True,
    },
    'ecd_donor': {
        'name_long': 'Expanded criteria donor',
        'type': 'donor',
        'na_values': ['U', 'PD', 'C'],
    },
    'hematocrit_don': {
        'name_long': 'Hematocrit donor',
        'type': 'donor'
    },
    'retransplant': {
        'name_long': 'Retransplant indiciation',
        'type': 'recipient'
    },
    'newpra': {
        'name_long': 'Combined panel-reactive antibody',
        'type': 'recipient'
    },
    'education': {
        'name_long': 'Education',
        'type': 'recipient',
        'na_values': [996, 998],
        'value_map': {
            1: 'grade school (0-8)',
            3: 'high school (9-12) or ged',
            4: 'attended college/technical school',
            5: 'associate/bachelor degree',
            6: 'post-college graduate degree',
            996: 'N/A',
            998: 'missing',
        },
    },
    'death': {
        'name_long': 'Patient died',
        'type': 'matching'
    },
    'deathr': {
        'name_long': 'Graft status',
        'type': 'matching'
    },
    'futd': {
        'name_long': 'Follow up time in days',
        'type': 'matching'
    },
    'height ratio': {
        'name_long': 'Height ratio',
        'type': 'matching',
        'na_values': ['#DIV/0!', 0],
    },
    'weight ratio': {
        'name_long': 'Weight ratio',
        'type': 'matching',
        'na_values': ['#DIV/0!', 0],
    },
    'congenital': {
        'name_long': 'Congenital heart disease indication',
        'type': 'recipient'
    },
    'cmassratio': {
        'name_long': 'Heart mass ratio',
        'type': 'matching'
    },
    'sexmatch': {
        'name_long': 'Sex match',
        'type': 'matching'
    },
    'anyecmo': {
        'name_long': 'Any ECMO support',
        'type': 'recipient'
    },
    'pvr': {
        'name_long': 'PVR',
        'type': 'recipient'
    },
    'anyvent': {
        'name_long': 'Any ventilation support',
        'type': 'recipient'
    },
    'tx_year': {
        'name_long': 'Transplant year',
        'type': 'matching',
        'na_values': [0, 1, 5, 'N', 999],
        'remove': True,
    },
}
