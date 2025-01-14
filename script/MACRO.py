# IDEA: Put here the hyperparameters that are used in the code common to all the runs
# (i.e., no need to track)
import pathlib

from dataclasses import dataclass

@dataclass
class MacroVariables:
    PATH = pathlib.Path("../course_findhr/for_students/data_notebooks/data")

    SUFFIX_DATASET = '1'  # '1' for demonstration, '2' for practice

    FILENAME_CURRICULA = "curricula{SUFFIX_DATASET}.csv"
    FILENAME_JOB_OFFERS = "job_offers{SUFFIX_DATASET}.csv"
    FILENAME_ADS_FAIR = 'score{SUFFIX_DATASET}_fair.csv'
    FILENAME_ADS_UNFAIR = 'score{SUFFIX_DATASET}_unfair.csv'

    FILENAME_FITNESS_MATRIX_FAIR = "fitness_mat{SUFFIX_DATASET}_fair.csv"
    FILENAME_FITNESS_MATRIX_UNFAIR = "fitness_mat{SUFFIX_DATASET}_unfair.csv"

    FILEPATH_CURRICULA = PATH / FILENAME_CURRICULA.format(SUFFIX_DATASET=SUFFIX_DATASET)
    FILEPATH_JOB_OFFERS = PATH / FILENAME_JOB_OFFERS.format(SUFFIX_DATASET=SUFFIX_DATASET)
    FILEPATH_ADS_FAIR = PATH / FILENAME_ADS_FAIR.format(SUFFIX_DATASET=SUFFIX_DATASET)
    FILEPATH_ADS_UNFAIR = PATH / FILENAME_ADS_UNFAIR.format(SUFFIX_DATASET=SUFFIX_DATASET)
    FILEPATH_FITNESS_MATRIX_FAIR = PATH / FILENAME_FITNESS_MATRIX_FAIR.format(SUFFIX_DATASET=SUFFIX_DATASET)
    FILEPATH_FITNESS_MATRIX_UNFAIR = PATH / FILENAME_FITNESS_MATRIX_UNFAIR.format(SUFFIX_DATASET=SUFFIX_DATASET)

    TOP_K = 10

    FAIR_DATA = True
