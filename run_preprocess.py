def run_preprocess():
    from utils.preprocess import ink2img_folder
    import os
    import pandas as pd

    data_folders = os.listdir("inkML_data")
    ink2img_folder([os.path.join("inkML_data", i) for i in ["TrainINKML_2013",
                                                            "trainData_2012_part1",
                                                            "trainData_2012_part2",
                                                            "CROHME_training_2011"]], "img_data")
    df = pd.read_csv("img_data/labels.csv")
    print(len(df), "images generated.")
    assert [i for i in os.listdir("img_data") if i.split('.png')[0][-1] != '0'] == ['labels.csv'], "Duplicate files in img_data folder. Please remove them and try again."
    img_files = set(os.listdir("img_data"))
    for i in df["name"]:
        assert i in img_files, f"{i} not found in img_data folder"
    assert len([i for i in df["label"] if '.ink' in i]) == 0, "Some labels are still in .ink format! The XML reading's probably corrupted. Did you add extra folders to the inkml dataset?"
    print("All checks passed.")

if __name__ == "__main__":
    run_preprocess()