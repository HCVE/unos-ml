heart_transplant_metadata = {
    'gender': {
        'name_long': 'Gender',
    },
    'abo': {
        'name_long': 'Blood group',
    },
    'wgt_kg_tcr': {
        'name_long': "Recipient's weight",
    },
    'hgt_cm_calc': {
        'name_long': 'Recipient\'s height calc',
    },
    'wgt_kg_calc': {
        'name_long': 'Recipient\'s weight calc',
    },
    'bmi_calc': {
        'name_long': 'BMI recipient',
    },
    'wgt_kg_don_calc': {
        'name_long': 'Donor\'s weight calc',
    },
    'hgt_cm_don_calc': {
        'name_long': 'Donor\'s height calc',
    },
    'organ': {
        'name_long': 'Organ',
    },
    'ebv_igg_cad_don': {
        'name_long': 'IgG for ebv infection in donor',
        'na_values': ['U'],
    },
    'iabp_tcr': {
        'name_long': 'Intra-aortic blood pump',
    },
    'inotropes_tcr': {
        'name_long': 'Inotropica',
    },
    'func_stat_tcr': {
        'name_long': 'Functional status at listing recipient',
    },
    'diab': {
        'name_long': 'Diabetes',
        'na_values': [998.],
    },
    'cereb_vasc': {
        'name_long': 'Cerebrovascular disease',
        'na_values': ['U'],
    },
    'most_rcnt_creat': {
        'name_long': 'Most recent creatinine (kidney function)',
    },
    'tot_serum_album': {
        'name_long': 'Serum albumine',
    },
    'impl_defibril': {
        'name_long': 'Defibrillator in recipient',
        'na_values': ['U'],
    },
    'hemo_sys_tcr': {
        'name_long': 'Systolic pulmonary pressure recipient (mmHg)',
    },
    'hemo_pa_dia_tcr': {
        'name_long': 'Diastolic pulmonary pressure recipient (mmHg)',
    },
    'hemo_pa_mn_tcr': {
        'name_long': 'Mean pulmonary pressure recipient (mmHg)',
    },
    'hemo_pcw_tcr': {
        'name_long': 'Mean pulmonary wedge pressure recipient (mmHg)',
    },
    'hemo_co_tcr': {
        'name_long': 'Cardiac output recipient (L/min)',
    },
    'cig_use': {
        'name_long': 'Cigarette use recipient',
    },
    'prior_card_surg_tcr': {
        'name_long': 'Prior cardiac surgery recipient',
        'na_values': ['U'],
    },
    'prior_card_surg_type_tcr': {
        'name_long': 'Prior cardiac surgery recipient type',
    },
    'ethcat': {
        'name_long': 'Ethnicity',
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
        'name_long': 'Recipient on ventilation',
        'na_values': ['U'],
    },
    'lvad ever': {
        'name_long': 'LV assist. device',
    },
    'rvad ever': {
        'name_long': 'RV assist. device',
    },
    'tah ever': {
        'name_long': 'Total artificial heart',
    },
    'med_cond_trr': {
        'name_long': 'medical condition—Recipient',
        'value_map': {
            1: 'ICU',
            2: 'hospital not ICU',
            3: 'not hospitalized'
        }
    },
    'ecmo_trr': {
        'name_long': 'ECM—Recipient',
    },
    'creat_trr': {
        'name_long': 'Creatinine—Recipient',
    },
    'dial_after_list': {
        'name_long': 'Dialysis after listing',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'tbili': {
        'name_long': 'Bilirubine—Recipient',
    },
    'transfusions': {
        'name_long': 'Blood transfusions since listing',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'hbv_core': {
        'name_long': 'Hepatitis B virus core',
        'na_values': ['U'],
        'value_map': {
            'N': 'negative',
            'ND': 'not done',
            'P': 'positive',
            'U': 'unknown'
        }
    },
    'hbv_sur_antigen': {
        'name_long': 'Hepatitis B virus core antigen',
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
        'na_values': ['U'],
        'value_map': {
            'N': 'negative',
            'ND': 'not done',
            'P': 'positive',
            'U': 'unknown'
        }
    },
    'cmv_igg': {
        'name_long': 'CMV IGG',
        'na_values': ['U', 'PD', 'C'],
        'value_map': {
            'N': 'negative',
            'ND': 'not done',
            'P': 'positive',
            'U': 'unknown'
        }
    },
    'cmv_igm': {
        'name_long': 'CMV IGM',
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
    },
    'gtime': {
        'name_long': 'Time of graft survival',
    },
    'pstatus': {
        'name_long': 'Time of graft survival',
    },
    'ptime': {
        'name_long': 'Time of patient survival',
    },
    'px_stat': {
        'name_long': 'Patient status',
    },
    'prev_tx': {
        'name_long': 'Previous transplantation',
    },
    'prev_tx_any': {
        'name_long': 'Previous transplantation any',
    },
    'hist_cocaine_don': {
        'name_long': 'History of cocaine use—Donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'age_don': {
        'name_long': 'Age—donor',
    },
    'ethcat_don': {
        'name_long': 'Ethnicity donor',
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
        'name_long': 'Hepatitis B virus core donor',
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
        'name_long': 'Hepatitis b virus surface antigen donor',
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
        'name_long': 'Bloodgroup',
    },
    'alcohol_heavy_don': {
        'name_long': 'Heavy alcohol use—Donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'gender_don': {
        'name_long': 'Gender—Donor',
    },
    'hep_c_anti_don': {
        'name_long': 'HCV serostatus—Donor',
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
        'name_long': 'non_hrt_don',
        'na_values': ['U'],
        'value_map': {
            'N': 'negative',
            'ND': 'not done',
            'P': 'positive',
            'U': 'unknown',
            'I': 'not clear'
        }
    },
    'blood_inf_don': {
        'name_long': 'Infection in blood source—Donor',
    },
    'bun_don': {
        'name_long': 'Blood urea nitrogen at transplantation in—Donor',
    },
    'creat_don': {
        'name_long': 'Creatinine at transplantation in—Donor',
    },
    'other_inf_don': {
        'name_long': 'Other infection at transplantation in—Donor',
    },
    'pulm_inf_don': {
        'name_long': 'Pulmonary infection at transplantation—Donor',
    },
    'sgot_don': {
        'name_long': 'AST at transplantation—Donor',
    },
    'sgpt_don': {
        'name_long': 'ALT at transplantation—Donor',
    },
    'tbili_don': {
        'name_long': 'Bilirubine at transplantation—Donor',
    },
    'urine_inf_don': {
        'name_long': 'Urinal infection at transplantation—Donor',
    },
    'vasodil_don': {
        'name_long': 'Vasodilatators at transplantation—Donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'vdrl_don': {
        'name_long': 'RPR-VDRL SEROLOGY at transplantation—Donor',
        'na_values': ['U', 'PD', 'C'],
        'value_map': {
            'subdivision N': 'negative',
            'ND': 'not done',
            'P': 'positive',
            'U': 'unknown'
        }
    },
    'clin_infect_don': {
        'name_long': 'Clinical infection at transplantation—Donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'contin_cocaine_don': {
        'name_long': 'Cocaine use in last 6 months—Donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'contin_oth_drug_don': {
        'name_long': 'Other drug use in last 6 months—Donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'extracranial_cancer_don': {
        'name_long': 'Extracranial cancer in donor at transplantation—Donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'hist_alcohol_old_don': {
        'name_long': 'History of alcohol in donor—Donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'cancer_site_don': {
        'name_long': 'Site of cancer—Donor',
        'value_map': {
            '1': 'no cancer'
        },
        'na_values': [998., 999.]
    },
    'hist_cig_don': {
        'name_long': 'History of cigarettes abuse—Donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes'
        }
    },
    'hist_hypertens_don': {
        'name_long': 'History of hypertension—Donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'hist_iv_drug_old_don': {
        'name_long': 'History of IV drugs—Donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'intracranial_cancer_don': {
        'name_long': 'Intracranial cancer at transplantation—Donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'hist_cancer_don': {
        'name_long': 'History of cancer—Donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'donor insulin': {
        'name_long': 'Insulin dependence—Donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'diabetes_don': {
        'name_long': 'Diabetes—Donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'hist_oth_drug_don': {
        'name_long': 'History of other drug abuse—Donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'skin_cancer_don': {
        'name_long': 'Skin cancer in donor at transplantation—Donor',
        'na_values': ['U'],
        'value_map': {
            'N': 'no',
            'Y': 'yes',
            'U': 'unknown'
        }
    },
    'cmv_don': {
        'name_long': 'CMV infection—Donor',
        'na_values': ['U', 'C'],
        'value_map': {
            'N': 'negative',
            'ND': 'not done',
            'P': 'positive',
            'U': 'unknown'
        }
    },
    'cod_cad_don': {
        'name_long': 'Cause of death—Donor',
        'na_values': [999.0],
        'value_map': {
            1: 'Anoxia',
            2: 'Cerebrovascular/Stroke',
            3: 'Head Trauma',
            4: 'Other',
        }
    },
    'death_mech_don': {
        'name_long': 'Cause of mechanical death—Donor',
        'na_values': [997., 995.]
    },
    'bmi_don_calc': {
        'name_long': 'Calculated BMI—Donor',
    },
    'multiorg': {
        'name_long': 'Multiorgan transplant',
    },
    'abo_mat': {
        'name_long': 'Matched bloodgroup',
        'na_values': [3],
    },
    'age': {
        'name_long': 'Age—Recipient',
    },
    'diag': {
        'name_long': 'Thoracic diagnosis recipient',
    },
    'dial_prior_tx': {
        'name_long': 'Dialysis prior to transplantation',
        'na_values': ['U'],
    },
    'ischtime': {
        'name_long': 'Ischemic time',
    },
    'life_sup_trr': {
        'name_long': 'Patient on life support—Recipient',
        'value_map': {
            'Y': 'yes',
            'N': 'no'
        },
    },
    'prior_card_surg_trr': {
        'name_long': 'Prior cardiac surgery between listing and transplantation',
        'na_values': ['U'],
        'value_map': {
            'Y': 'yes',
            'N': 'no'
        },
    },
    'malig': {
        'name_long': 'Any previous cancers',
        'na_values': ['U'],
        'value_map': {
            'Y': 'yes',
            'N': 'no',
            'U': 'unknown'
        },
    },
    'distance': {
        'name_long': 'Distance between donor hospital and transplantation center',
    },
    'vent_support_after_list': {
        'name_long': 'Ventilation support after listing',
        'na_values': ['U'],
        'value_map': {
            'Y': 'yes',
            'N': 'no',
            'U': 'unknown'
        },
    },
    'protein_urine': {
        'name_long': 'Proteinuria',
        'na_values': ['U'],
        'value_map': {
            'Y': 'yes',
            'N': 'no',
            'U': 'unknown'
        },
    },
    'hist_mi': {
        'name_long': 'History of previous myocardial infarct',
        'na_values': ['U'],
        'value_map': {
            'Y': 'yes',
            'N': 'no',
            'U': 'unknown'
        },
    },
    'lv_eject_meth': {
        'name_long': 'Method measuring left ventricle ejection',
    },
    'lv_eject': {
        'name_long': 'Left ventricle ejection—Donor',
    },
    'coronary_angio': {
        'name_long': 'Angiogram in donorheart—Donor',
        'value_map': {
            1: 'no angio',
            2: 'normal angio',
            3: 'abnormal angio'
        },
    },
    'vessels_50sten': {
        'name_long': 'Number of vessels with 50% stenosis in donor heart—Donor',
    },
    'biopsy_dgn': {
        'name_long': 'Heart biopsy—Donor',
    },
    'tattoos': {
        'name_long': 'Tattoos—Donor',
        'na_values': ['U'],
        'value_map': {
            'Y': 'yes',
            'N': 'no',
            'U': 'unknown'
        },
    },
    'cdc_risk_hiv_don': {
        'name_long': 'CDC risk factors for blood bourne disease—Donor',
        'na_values': ['U', 'PD', 'C'],
        'value_map': {
            'N': 'negative',
            'ND': 'not done',
            'P': 'positive',
            'U': 'unknown'
        },
    },
    'ecd_donor': {
        'name_long': 'Expanded Donor Including Brain Dead and DCD—Donor',
        'na_values': ['U', 'PD', 'C'],
    },
    'hematocrit_don': {
        'name_long': 'Hematocrit—Donor',
    },
    'retransplant': {
        'name_long': 'Retransplant',
    },
    'newpra': {
        'name_long': 'Combined panel-reactive antibody',
    },
    'education': {
        'name_long': 'Education',
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
    },
    'deathr': {
        'name_long': 'Graft status',
    },
    'futd': {
        'name_long': 'Follow up time in days',
    },
    'height ratio': {
        'name_long': 'Height ratio',
        'na_values': ['#DIV/0!', 0],
    },
    'weight ratio': {
        'name_long': 'Weight ratio',
        'na_values': ['#DIV/0!', 0],
    },
    'congenital': {
        'name_long': 'Congenital heart disease diagnosis',
    },
    'cmassratio': {
        'name_long': 'cmassratio',
    },
}
