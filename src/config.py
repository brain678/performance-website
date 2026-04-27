from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Dataset.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"

TARGET_RISK = "at_risk"
TARGET_PERF = "cgpa"

ID_COLUMNS = ["name", "matric_no"]

REQUIRED_COLUMNS = [
	"name",
	"matric_no",
	"department",
	"level",
	"attendance",
	"study_hours",
	"ca1",
	"ca2",
	"assignment_score",
	"num_courses",
	"exam_score",
	"gpa_sem1",
	"gpa_sem2",
	"cgpa",
	"avg_course_score",
]

# Drop to reduce leakage when predicting at-risk status.
RISK_DROP_COLUMNS = ["cgpa"]

# Stricter leakage controls for more realistic risk evaluation.
STRICT_RISK_DROP_COLUMNS = [
	"cgpa",
	"gpa_sem1",
	"gpa_sem2",
	"avg_course_score",
]

STRICT_GROUP_COLUMN = "matric_no"

# Add columns here if you want to exclude them for performance prediction.
PERF_DROP_COLUMNS = []

CATEGORICAL_COLUMNS = ["department", "level"]

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
EXPLAIN_MAX_SAMPLES = 200
PLOTS_DIR_NAME = "plots"
TOP_FEATURES = 20
TUNE_ITER = 25
RISK_THRESHOLD_DEFAULT = 0.5
RISK_VAL_SIZE = 0.2
RISK_TIER_THRESHOLDS = [0.33, 0.66]
RISK_TIER_LABELS = ["low", "medium", "high"]

# Used only when at_risk is missing and must be generated.
AT_RISK_CGPA_THRESHOLD = 2.0
AT_RISK_AVG_SCORE_THRESHOLD = 45.0
AT_RISK_ATTENDANCE_THRESHOLD = 50.0
