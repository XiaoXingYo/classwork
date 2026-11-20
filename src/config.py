"""Project-wide configuration and paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "data"
OUT_DIR = ROOT / "outputs"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
LOG_DIR = OUT_DIR / "logs"
REPORT_DIR = ROOT / "report"

for _p in (OUT_DIR, TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_DIR):
    _p.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 2024, 7, 13, 99, 2026, 11, 18, 123, 777]
IRIS_SEEDS = list(SEEDS)

IRIS_FEATURES = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]
IRIS_LABEL = "Species"

# MEDV is the response variable and is excluded from the unsupervised feature space.
HOUSING_FEATURES = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
    "RAD", "TAX", "PTRATIO", "B", "LSTAT",
]
HOUSING_LABEL = "MEDV"

FEATURE_LABELS = {
    "Sepal_Length": "花萼长度",
    "Sepal_Width": "花萼宽度",
    "Petal_Length": "花瓣长度",
    "Petal_Width": "花瓣宽度",
    "Species": "鸢尾花类别",
    "CRIM": "城镇人均犯罪率",
    "ZN": "住宅用地比例",
    "INDUS": "非零售商业用地比例",
    "CHAS": "是否临查尔斯河",
    "NOX": "一氧化氮浓度",
    "RM": "每户平均房间数",
    "AGE": "1940年前建成房屋比例",
    "DIS": "到就业中心加权距离",
    "RAD": "高速公路可达性指数",
    "TAX": "财产税率",
    "PTRATIO": "师生比",
    "B": "黑人比例变换值",
    "LSTAT": "低收入人口比例",
    "MEDV": "自住房屋中位价（千美元）",
    "Petal_Area": "花瓣面积",
    "Sepal_Area": "花萼面积",
    "Petal_Aspect": "花瓣长宽比",
    "Sepal_Aspect": "花萼长宽比",
    "Petal_Sepal_Ratio": "花瓣/花萼长度比",
    "CRIM_log": "犯罪率对数",
    "LSTAT_log": "低收入比例对数",
    "TAX_log": "税率对数",
    "RM2": "房间数平方",
    "LSTAT_RM": "低收入比例/房间数",
    "NOX_DIS": "污染×距离",
    "TAX_RAD": "税率/可达性",
    "AGE_LSTAT": "房龄×低收入比例",
    "CRIM_LSTAT": "犯罪率×低收入比例",
    "PTRATIO_LSTAT": "师生比×低收入比例",
}
