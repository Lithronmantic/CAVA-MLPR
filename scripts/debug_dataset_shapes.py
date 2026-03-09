import yaml
from dataset import AVFromCSV

cfg = yaml.safe_load(open("./configs/selfsup_opt.yaml ", "r", encoding="utf-8"))

root = cfg["data"].get("root", ".")
csv_path = cfg["data"]["val_csv"]  # 或 test_csv
C = cfg["data"]["num_classes"]
class_names = cfg["data"]["class_names"]
vcfg = cfg["video"]
acfg = cfg["audio"]

ds = AVFromCSV(csv_path, root, C, class_names, vcfg, acfg, is_unlabeled=False)

x = ds[0]
print("type(x) =", type(x))

if isinstance(x, dict):
    print("keys =", x.keys())
    v = x["video"]
    a = x["audio"]
    y = x["label"]
else:
    v, a, y = x[:3]

print("video shape =", getattr(v, "shape", None), "dtype =", getattr(v, "dtype", None))
print("audio shape =", getattr(a, "shape", None), "dtype =", getattr(a, "dtype", None))
print("label =", y)
